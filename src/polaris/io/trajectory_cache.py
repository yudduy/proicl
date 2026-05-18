from __future__ import annotations

import datetime as _dt
import hashlib
import json
import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TrajectoryKey:
    model_id: str
    track: str
    problem_id: str
    prompt_id: str
    sample_idx: int
    alpha: float
    seed: int


@dataclass(frozen=True)
class TrajectoryRecord:
    key: TrajectoryKey
    prompt_hash: str
    generation: str
    response_contains_prompt: bool
    token_ids: list[int]
    logprobs_norm: list[float]
    logprobs_unnorm: list[float]
    acceptance_ratio: float | None
    prompt_token_count: int
    generation_token_count: int
    wall_clock_seconds: float
    dollar_cost: float
    verifier_result: dict[str, Any] | None = None
    created_at: str | None = None

    @property
    def verifier_passed(self) -> bool | None:
        if self.verifier_result is None:
            return None
        return bool(self.verifier_result.get("passed", False))


class TrajectoryCache:
    """SQLite cache for idempotent base-model trajectory reuse.

    One row is one generated candidate trajectory. The key intentionally
    includes track/problem/prompt/sample/alpha/seed so interrupted spot runs can
    resume by querying missing cells instead of re-generating completed work.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "TrajectoryCache":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @staticmethod
    def prompt_hash(prompt_text: str) -> str:
        return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()

    def get(
        self,
        *,
        model_id: str,
        track: str,
        problem_id: str,
        prompt_id: str,
        sample_idx: int,
        alpha: float,
        seed: int,
    ) -> TrajectoryRecord | None:
        row = self._conn.execute(
            """
            SELECT * FROM trajectories
            WHERE model_id = ?
              AND track = ?
              AND problem_id = ?
              AND prompt_id = ?
              AND sample_idx = ?
              AND alpha = ?
              AND seed = ?
            """,
            (model_id, track, problem_id, prompt_id, sample_idx, alpha, seed),
        ).fetchone()
        return _row_to_record(row) if row is not None else None

    def put(self, record: TrajectoryRecord, *, overwrite: bool = False) -> None:
        now = record.created_at or _utc_now()
        values = (
            record.key.model_id,
            record.key.track,
            record.key.problem_id,
            record.key.prompt_id,
            record.key.sample_idx,
            record.key.alpha,
            record.key.seed,
            record.prompt_hash,
            record.generation,
            int(record.response_contains_prompt),
            json.dumps(record.token_ids, separators=(",", ":")),
            _pack_f16(record.logprobs_norm),
            _pack_f16(record.logprobs_unnorm),
            record.acceptance_ratio,
            record.prompt_token_count,
            record.generation_token_count,
            record.wall_clock_seconds,
            record.dollar_cost,
            json.dumps(record.verifier_result, ensure_ascii=False)
            if record.verifier_result is not None
            else None,
            int(record.verifier_passed) if record.verifier_passed is not None else None,
            now,
        )
        try:
            if overwrite:
                self._conn.execute(
                    """
                    INSERT INTO trajectories (
                      model_id, track, problem_id, prompt_id, sample_idx, alpha, seed,
                      prompt_hash, generation, response_contains_prompt, token_ids,
                      logprobs_norm, logprobs_unnorm, acceptance_ratio,
                      prompt_token_count, generation_token_count, wall_clock_seconds,
                      dollar_cost, verifier_result, verifier_passed, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(model_id, track, problem_id, prompt_id, sample_idx, alpha, seed)
                    DO UPDATE SET
                      prompt_hash = excluded.prompt_hash,
                      generation = excluded.generation,
                      response_contains_prompt = excluded.response_contains_prompt,
                      token_ids = excluded.token_ids,
                      logprobs_norm = excluded.logprobs_norm,
                      logprobs_unnorm = excluded.logprobs_unnorm,
                      acceptance_ratio = excluded.acceptance_ratio,
                      prompt_token_count = excluded.prompt_token_count,
                      generation_token_count = excluded.generation_token_count,
                      wall_clock_seconds = excluded.wall_clock_seconds,
                      dollar_cost = excluded.dollar_cost,
                      verifier_result = excluded.verifier_result,
                      verifier_passed = excluded.verifier_passed,
                      created_at = excluded.created_at
                    """,
                    values,
                )
            else:
                self._conn.execute(
                    """
                    INSERT INTO trajectories (
                      model_id, track, problem_id, prompt_id, sample_idx, alpha, seed,
                      prompt_hash, generation, response_contains_prompt, token_ids,
                      logprobs_norm, logprobs_unnorm, acceptance_ratio,
                      prompt_token_count, generation_token_count, wall_clock_seconds,
                      dollar_cost, verifier_result, verifier_passed, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    values,
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"trajectory already cached for key: {record.key}") from exc
        self._conn.commit()

    def mark_verified(
        self,
        key: TrajectoryKey,
        *,
        verifier_result: dict[str, Any],
    ) -> None:
        cur = self._conn.execute(
            """
            UPDATE trajectories
            SET verifier_result = ?, verifier_passed = ?
            WHERE model_id = ?
              AND track = ?
              AND problem_id = ?
              AND prompt_id = ?
              AND sample_idx = ?
              AND alpha = ?
              AND seed = ?
            """,
            (
                json.dumps(verifier_result, ensure_ascii=False),
                int(bool(verifier_result.get("passed", False))),
                key.model_id,
                key.track,
                key.problem_id,
                key.prompt_id,
                key.sample_idx,
                key.alpha,
                key.seed,
            ),
        )
        if cur.rowcount != 1:
            raise KeyError(f"trajectory not found for key: {key}")
        self._conn.commit()

    def iter_for_prompt(
        self,
        *,
        model_id: str,
        track: str,
        problem_id: str,
        prompt_id: str,
        seed: int | None = None,
    ) -> Iterable[TrajectoryRecord]:
        sql = """
            SELECT * FROM trajectories
            WHERE model_id = ?
              AND track = ?
              AND problem_id = ?
              AND prompt_id = ?
        """
        args: list[Any] = [model_id, track, problem_id, prompt_id]
        if seed is not None:
            sql += " AND seed = ?"
            args.append(seed)
        sql += " ORDER BY alpha, sample_idx"
        for row in self._conn.execute(sql, args):
            yield _row_to_record(row)

    def _init_schema(self) -> None:
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache_meta (
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trajectories (
              model_id TEXT NOT NULL,
              track TEXT NOT NULL,
              problem_id TEXT NOT NULL,
              prompt_id TEXT NOT NULL,
              sample_idx INTEGER NOT NULL,
              alpha REAL NOT NULL,
              seed INTEGER NOT NULL,
              prompt_hash TEXT NOT NULL,
              generation TEXT NOT NULL,
              response_contains_prompt INTEGER NOT NULL,
              token_ids TEXT NOT NULL,
              logprobs_norm BLOB NOT NULL,
              logprobs_unnorm BLOB NOT NULL,
              acceptance_ratio REAL,
              prompt_token_count INTEGER NOT NULL,
              generation_token_count INTEGER NOT NULL,
              wall_clock_seconds REAL NOT NULL,
              dollar_cost REAL NOT NULL,
              verifier_result TEXT,
              verifier_passed INTEGER,
              created_at TEXT NOT NULL,
              PRIMARY KEY (
                model_id, track, problem_id, prompt_id, sample_idx, alpha, seed
              )
            )
            """
        )
        self._conn.execute(
            """
            INSERT INTO cache_meta(key, value)
            VALUES ('schema_version', ?)
            ON CONFLICT(key) DO NOTHING
            """,
            (str(SCHEMA_VERSION),),
        )
        self._conn.commit()


def _row_to_record(row: sqlite3.Row) -> TrajectoryRecord:
    verifier_raw = row["verifier_result"]
    return TrajectoryRecord(
        key=TrajectoryKey(
            model_id=row["model_id"],
            track=row["track"],
            problem_id=row["problem_id"],
            prompt_id=row["prompt_id"],
            sample_idx=int(row["sample_idx"]),
            alpha=float(row["alpha"]),
            seed=int(row["seed"]),
        ),
        prompt_hash=row["prompt_hash"],
        generation=row["generation"],
        response_contains_prompt=bool(row["response_contains_prompt"]),
        token_ids=json.loads(row["token_ids"]),
        logprobs_norm=_unpack_f16(row["logprobs_norm"]),
        logprobs_unnorm=_unpack_f16(row["logprobs_unnorm"]),
        acceptance_ratio=row["acceptance_ratio"],
        prompt_token_count=int(row["prompt_token_count"]),
        generation_token_count=int(row["generation_token_count"]),
        wall_clock_seconds=float(row["wall_clock_seconds"]),
        dollar_cost=float(row["dollar_cost"]),
        verifier_result=json.loads(verifier_raw) if verifier_raw else None,
        created_at=row["created_at"],
    )


def _pack_f16(values: list[float]) -> bytes:
    if not values:
        return b""
    return struct.pack(f"<{len(values)}e", *values)


def _unpack_f16(blob: bytes) -> list[float]:
    if not blob:
        return []
    n = len(blob) // 2
    return [float(x) for x in struct.unpack(f"<{n}e", blob)]


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")
