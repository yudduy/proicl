from __future__ import annotations

import datetime as _dt
import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from polaris.core.memory import MemoryEntry


class PersistentMemoryLedger:
    """SQLite-backed memory ledger for production runs.

    The in-memory `MemoryStore` remains the fast inference object. This ledger is
    the durable audit trail: admissions, rejections, retrieval eligibility,
    posterior updates, snapshots, and pruning decisions.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "PersistentMemoryLedger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def admit(
        self,
        entry: MemoryEntry,
        *,
        track: str,
        verifier_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO memory_entries (
              id, track, archive_prompt_id, descriptor, strategy_text, token_count,
              source_query_id, reliability_alpha, reliability_beta, verifier_id,
              metadata_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
              reliability_alpha = excluded.reliability_alpha,
              reliability_beta = excluded.reliability_beta,
              metadata_json = excluded.metadata_json
            """,
            (
                entry.id,
                track,
                entry.archive_prompt_id,
                entry.descriptor,
                entry.strategy_text,
                entry.token_count,
                entry.source_query_id,
                entry.reliability_alpha,
                entry.reliability_beta,
                verifier_id,
                json.dumps(metadata or {}, sort_keys=True),
                _utc_now(),
            ),
        )
        self.record_event(
            event_type="admission",
            entry_ids=[entry.id],
            query_id=entry.source_query_id,
            payload={"verifier_id": verifier_id, "metadata": metadata or {}},
        )
        self._conn.commit()

    def reject(
        self,
        *,
        candidate_trace_id: str,
        query_id: str,
        reason: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.record_event(
            event_type="rejection",
            entry_ids=[],
            query_id=query_id,
            payload={
                "candidate_trace_id": candidate_trace_id,
                "reason": reason,
                "metadata": metadata or {},
            },
        )
        self._conn.commit()

    def record_retrieval(
        self,
        *,
        query_id: str,
        eligible_ids: Iterable[str],
        retrieved_ids: Iterable[str],
        verifier_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.record_event(
            event_type="retrieval",
            entry_ids=list(retrieved_ids),
            query_id=query_id,
            payload={
                "eligible_ids": list(eligible_ids),
                "retrieved_ids": list(retrieved_ids),
                "verifier_metadata": verifier_metadata or {},
            },
        )
        self._conn.commit()

    def update_posterior(
        self,
        entry_ids: Iterable[str],
        *,
        verifier_outcome: int,
        query_id: str | None = None,
    ) -> None:
        if verifier_outcome not in (0, 1):
            raise ValueError("verifier_outcome must be 0 or 1")
        entry_ids = list(entry_ids)
        for entry_id in entry_ids:
            self._conn.execute(
                """
                UPDATE memory_entries
                SET reliability_alpha = reliability_alpha + ?,
                    reliability_beta = reliability_beta + ?
                WHERE id = ?
                """,
                (verifier_outcome, 1 - verifier_outcome, entry_id),
            )
        self.record_event(
            event_type="posterior_update",
            entry_ids=entry_ids,
            query_id=query_id,
            payload={"verifier_outcome": verifier_outcome},
        )
        self._conn.commit()

    def rollback_incomplete_queries(
        self,
        completed_query_ids: Iterable[str],
        *,
        expected_query_ids: Iterable[str] | None = None,
    ) -> int:
        """Remove query-scoped memory updates that do not have a selected row.

        Evaluation checkpoints treat a row in ``selected.jsonl`` as the commit
        marker for a problem. If a process dies after mutating the memory ledger
        but before writing that selected row, replay should not inherit those
        partial retrieval/posterior/admission side effects.
        """

        completed = {str(query_id) for query_id in completed_query_ids}
        expected = (
            {str(query_id) for query_id in expected_query_ids}
            if expected_query_ids is not None
            else None
        )
        rows = [
            dict(row)
            for row in self._conn.execute(
                """
                SELECT id, event_type, entry_ids_json, query_id, payload_json
                FROM memory_events
                WHERE query_id IS NOT NULL
                ORDER BY id DESC
                """
            )
        ]
        rolled_back = 0
        for row in rows:
            query_id = str(row["query_id"])
            if query_id in completed:
                continue
            if expected is not None and query_id not in expected:
                continue
            entry_ids = json.loads(row["entry_ids_json"])
            if row["event_type"] == "posterior_update":
                payload = json.loads(row["payload_json"])
                outcome = int(payload.get("verifier_outcome", 0))
                for entry_id in entry_ids:
                    self._conn.execute(
                        """
                        UPDATE memory_entries
                        SET reliability_alpha = reliability_alpha - ?,
                            reliability_beta = reliability_beta - ?
                        WHERE id = ?
                        """,
                        (outcome, 1 - outcome, entry_id),
                    )
            elif row["event_type"] == "admission":
                for entry_id in entry_ids:
                    self._conn.execute(
                        "DELETE FROM memory_entries WHERE id = ?",
                        (entry_id,),
                    )
            self._conn.execute("DELETE FROM memory_events WHERE id = ?", (row["id"],))
            rolled_back += 1
        self._conn.commit()
        return rolled_back

    def snapshot_posteriors(self, *, label: str) -> list[dict[str, Any]]:
        rows = [dict(row) for row in self._conn.execute("SELECT * FROM memory_entries")]
        self._conn.execute(
            """
            INSERT INTO posterior_snapshots (label, payload_json, created_at)
            VALUES (?, ?, ?)
            """,
            (label, json.dumps(rows, sort_keys=True), _utc_now()),
        )
        self._conn.commit()
        return rows

    def prune(self, *, max_entries_per_prompt: int) -> list[str]:
        pruned: list[str] = []
        prompts = [
            row["archive_prompt_id"]
            for row in self._conn.execute(
                "SELECT DISTINCT archive_prompt_id FROM memory_entries"
            )
        ]
        for prompt_id in prompts:
            rows = [
                dict(row)
                for row in self._conn.execute(
                    """
                    SELECT id, reliability_alpha, reliability_beta
                    FROM memory_entries
                    WHERE archive_prompt_id = ?
                    """,
                    (prompt_id,),
                )
            ]
            rows.sort(
                key=lambda row: row["reliability_alpha"]
                / (row["reliability_alpha"] + row["reliability_beta"]),
                reverse=True,
            )
            drop = rows[max_entries_per_prompt:]
            for row in drop:
                pruned.append(row["id"])
                self._conn.execute("DELETE FROM memory_entries WHERE id = ?", (row["id"],))
        if pruned:
            self.record_event(
                event_type="prune",
                entry_ids=pruned,
                query_id=None,
                payload={"max_entries_per_prompt": max_entries_per_prompt},
            )
        self._conn.commit()
        return pruned

    def entries(self) -> list[MemoryEntry]:
        rows = self._conn.execute(
            """
            SELECT id, archive_prompt_id, descriptor, strategy_text, token_count,
                   source_query_id, reliability_alpha, reliability_beta
            FROM memory_entries
            ORDER BY id
            """
        )
        return [
            MemoryEntry(
                id=row["id"],
                archive_prompt_id=row["archive_prompt_id"],
                descriptor=row["descriptor"],
                strategy_text=row["strategy_text"],
                token_count=row["token_count"],
                source_query_id=row["source_query_id"],
                reliability_alpha=row["reliability_alpha"],
                reliability_beta=row["reliability_beta"],
            )
            for row in rows
        ]

    def events(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self._conn.execute("SELECT * FROM memory_events")]

    def record_event(
        self,
        *,
        event_type: str,
        entry_ids: Iterable[str],
        query_id: str | None,
        payload: dict[str, Any],
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO memory_events (
              event_type, entry_ids_json, query_id, payload_json, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                event_type,
                json.dumps(list(entry_ids), sort_keys=True),
                query_id,
                json.dumps(payload, sort_keys=True),
                _utc_now(),
            ),
        )

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_entries (
              id TEXT PRIMARY KEY,
              track TEXT NOT NULL,
              archive_prompt_id TEXT NOT NULL,
              descriptor TEXT NOT NULL,
              strategy_text TEXT NOT NULL,
              token_count INTEGER NOT NULL,
              source_query_id TEXT NOT NULL,
              reliability_alpha REAL NOT NULL,
              reliability_beta REAL NOT NULL,
              verifier_id TEXT NOT NULL,
              metadata_json TEXT NOT NULL,
              created_at TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_events (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              event_type TEXT NOT NULL,
              entry_ids_json TEXT NOT NULL,
              query_id TEXT,
              payload_json TEXT NOT NULL,
              created_at TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS posterior_snapshots (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              label TEXT NOT NULL,
              payload_json TEXT NOT NULL,
              created_at TEXT NOT NULL
            )
            """
        )
        self._conn.commit()


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")
