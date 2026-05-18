import base64
import gc
import io
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tarfile

MODEL_ID = os.environ.get("POLARIS_CALIBRATION_MODEL_ID", "Qwen/Qwen2.5-Math-7B")
MODEL_REVISION = os.environ.get("POLARIS_CALIBRATION_MODEL_REVISION") or None
TEMP = float(os.environ.get("POLARIS_CALIBRATION_TEMPERATURE", "0.25"))
HF_DTYPE = os.environ.get("POLARIS_CALIBRATION_HF_DTYPE", "float32")
HF_DEVICE_MAP_AUTO = os.environ.get("POLARIS_CALIBRATION_HF_DEVICE_MAP_AUTO", "0") == "1"
HF_ATTN_IMPLEMENTATION = os.environ.get("POLARIS_CALIBRATION_HF_ATTN_IMPLEMENTATION") or None
HF_SCORING_MODE = os.environ.get("POLARIS_CALIBRATION_HF_SCORING_MODE", "forward")
VLLM_DTYPE = os.environ.get("POLARIS_CALIBRATION_VLLM_DTYPE", "float32")
VLLM_MODEL_IMPL = os.environ.get("POLARIS_CALIBRATION_VLLM_MODEL_IMPL", "transformers")
VLLM_ENFORCE_EAGER = os.environ.get("POLARIS_CALIBRATION_VLLM_ENFORCE_EAGER", "1") == "1"
VLLM_DISABLE_ASYNC_OUTPUT_PROC = (
    os.environ.get("POLARIS_CALIBRATION_VLLM_DISABLE_ASYNC_OUTPUT_PROC", "1") == "1"
)
VLLM_ENABLE_PREFIX_CACHING = (
    os.environ.get("POLARIS_CALIBRATION_VLLM_ENABLE_PREFIX_CACHING", "1") == "1"
)
VLLM_RESET_PREFIX_CACHE_FOR_SCORING = (
    os.environ.get("POLARIS_CALIBRATION_VLLM_RESET_PREFIX_CACHE_FOR_SCORING", "1")
    == "1"
)
EMIT_TGZ = os.environ.get("POLARIS_CALIBRATION_EMIT_TGZ", "1") == "1"
SEGMENT_LENS = [1, 2, 8, 32, 128]
ATOL = 1e-3
AMBIG_LIMIT = 0.01

out = Path(os.environ["OUT"])
cal = out / "calibration"
cal.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("VLLM_USE_V1", "0")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch
import torch.nn.functional as F
import transformers
from huggingface_hub import snapshot_download

def hf_torch_dtype():
    if HF_DTYPE == "auto":
        return "auto"
    if HF_DTYPE == "float32":
        return torch.float32
    if HF_DTYPE == "bfloat16":
        return torch.bfloat16
    if HF_DTYPE == "float16":
        return torch.float16
    raise SystemExit(
        "POLARIS_CALIBRATION_HF_DTYPE must be one of "
        "auto,float32,bfloat16,float16"
    )

model_path = snapshot_download(MODEL_ID, revision=MODEL_REVISION)
tok_hf = transformers.AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
tok_vllm = transformers.AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
if tok_hf.pad_token_id is None:
    tok_hf.pad_token_id = tok_hf.eos_token_id
if tok_vllm.pad_token_id is None:
    tok_vllm.pad_token_id = tok_vllm.eos_token_id

prompt = (
    "Solve the problem and put the final answer in boxed form.\n"
    "Problem: Convert the point $(0,3)$ in rectangular coordinates to polar "
    "coordinates. Enter your answer as $(r,\\theta)$ with $r > 0$.\n"
    "Solution:"
)
target_text = " The radius is 3 and the angle is pi/2, so the final answer is boxed."
prefix_hf = tok_hf.encode(prompt)
prefix_vllm = tok_vllm.encode(prompt)
target_hf = tok_hf.encode(target_text, add_special_tokens=False)
target_vllm = tok_vllm.encode(target_text, add_special_tokens=False)
eos_id = tok_hf.eos_token_id if tok_hf.eos_token_id is not None else target_hf[0]

def extend(ids, n, fallback):
    row = list(ids) or [fallback]
    while len(row) < n:
        row.extend(ids or [fallback])
    return row[:n]

segments_hf = [extend(target_hf, n, eos_id) for n in SEGMENT_LENS]
segments_vllm = [extend(target_vllm, n, eos_id) for n in SEGMENT_LENS]
segments_hf.append([eos_id])
segments_vllm.append([eos_id])
segments_hf.append([target_hf[0]] * 8)
segments_vllm.append([target_vllm[0]] * 8)
tokenizer_parity = {
    "passed": prefix_hf == prefix_vllm and segments_hf == segments_vllm,
    "prefix_len": len(prefix_hf),
    "segment_lens": [len(x) for x in segments_hf],
    "model_id": MODEL_ID,
    "model_revision": MODEL_REVISION,
    "tokenizer_revision": MODEL_REVISION,
    "special_tokens": {
        "hf_pad_token_id": tok_hf.pad_token_id,
        "hf_eos_token_id": tok_hf.eos_token_id,
        "vllm_pad_token_id": tok_vllm.pad_token_id,
        "vllm_eos_token_id": tok_vllm.eos_token_id,
    },
}
if not tokenizer_parity["passed"]:
    raise SystemExit("tokenizer parity failed")

def hf_score(prefix_ids, target_ids):
    if HF_SCORING_MODE == "cached_decode":
        return hf_score_cached_decode(prefix_ids, target_ids)
    if HF_SCORING_MODE == "forced_generate":
        return hf_score_forced_generate(prefix_ids, target_ids)
    if HF_SCORING_MODE != "forward":
        raise SystemExit(
            "POLARIS_CALIBRATION_HF_SCORING_MODE must be one of "
            "forward,cached_decode,forced_generate"
        )
    full = list(prefix_ids) + list(target_ids)
    device = next(hf_model.parameters()).device
    input_ids = torch.tensor([full], dtype=torch.long, device=device)
    attn = torch.ones_like(input_ids)
    with torch.no_grad():
        outp = hf_model(input_ids=input_ids, attention_mask=attn, use_cache=False)
    rows = outp.logits[0, len(prefix_ids) - 1 : len(full) - 1, :].float()
    targets = torch.tensor(target_ids, dtype=torch.long, device=rows.device)
    norm = F.log_softmax(rows / TEMP, dim=-1).gather(-1, targets[:, None]).squeeze(-1)
    unnorm = ((1.0 / TEMP) * F.log_softmax(rows, dim=-1)).gather(-1, targets[:, None]).squeeze(-1)
    norm_list = [float(x) for x in norm.detach().cpu().tolist()]
    unnorm_list = [float(x) for x in unnorm.detach().cpu().tolist()]
    del outp, input_ids, attn, rows, targets, norm, unnorm
    return {"lp_norm": sum(norm_list), "lp_unnorm": sum(unnorm_list), "lp_norm_tokens": norm_list, "lp_unnorm_tokens": unnorm_list}

def hf_score_cached_decode(prefix_ids, target_ids):
    if not target_ids:
        return {"lp_norm": 0.0, "lp_unnorm": 0.0, "lp_norm_tokens": [], "lp_unnorm_tokens": []}
    device = next(hf_model.parameters()).device
    current = torch.tensor([list(prefix_ids)], dtype=torch.long, device=device)
    attention_mask = torch.ones((1, len(prefix_ids)), dtype=torch.long, device=device)
    past = None
    norm_list = []
    unnorm_list = []
    for pos, target_id in enumerate(target_ids):
        with torch.no_grad():
            outp = hf_model(
                input_ids=current,
                attention_mask=attention_mask,
                past_key_values=past,
                use_cache=True,
            )
        row = outp.logits[0, -1, :].float()
        norm_list.append(float(F.log_softmax(row / TEMP, dim=-1)[int(target_id)].detach().cpu()))
        unnorm_list.append(float(((1.0 / TEMP) * F.log_softmax(row, dim=-1)[int(target_id)]).detach().cpu()))
        past = outp.past_key_values
        current = torch.tensor([[int(target_id)]], dtype=torch.long, device=device)
        attention_mask = torch.ones((1, len(prefix_ids) + pos + 1), dtype=torch.long, device=device)
        del outp, row
    return {"lp_norm": sum(norm_list), "lp_unnorm": sum(unnorm_list), "lp_norm_tokens": norm_list, "lp_unnorm_tokens": unnorm_list}

class HFForcedTokenProcessor:
    def __init__(self, target_ids):
        self.target_ids = [int(x) for x in target_ids]

    def __call__(self, input_ids, scores):
        pos = input_ids.shape[-1] - self.prefix_len
        if pos >= len(self.target_ids):
            return scores
        forced = torch.full_like(scores, -torch.inf)
        forced[:, self.target_ids[pos]] = 0.0
        return forced

def hf_score_forced_generate(prefix_ids, target_ids):
    input_ids = torch.tensor([list(prefix_ids)], dtype=torch.long, device=next(hf_model.parameters()).device)
    processor = HFForcedTokenProcessor(target_ids)
    processor.prefix_len = len(prefix_ids)
    with torch.no_grad():
        outp = hf_model.generate(
            input_ids=input_ids,
            max_new_tokens=len(target_ids),
            min_new_tokens=0,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            output_logits=True,
            eos_token_id=tok_hf.eos_token_id,
            pad_token_id=tok_hf.pad_token_id,
            logits_processor=[processor],
        )
    observed = [int(x) for x in outp.sequences[0][len(prefix_ids):].detach().cpu().tolist()]
    if observed != list(target_ids):
        raise RuntimeError(f"HF forced ids mismatch expected={target_ids[:8]} observed={observed[:8]}")
    unscaled = torch.stack(outp.logits, dim=0).squeeze(1).float()
    scaled = torch.stack(outp.scores, dim=0).squeeze(1).float()
    targets = torch.tensor(target_ids, dtype=torch.long, device=unscaled.device)
    norm = F.log_softmax(scaled, dim=-1).gather(-1, targets[:, None]).squeeze(-1)
    unnorm = ((1.0 / TEMP) * F.log_softmax(unscaled, dim=-1)).gather(-1, targets[:, None]).squeeze(-1)
    norm_list = [float(x) for x in norm.detach().cpu().tolist()]
    unnorm_list = [float(x) for x in unnorm.detach().cpu().tolist()]
    del outp, input_ids, unscaled, scaled, targets, norm, unnorm
    return {"lp_norm": sum(norm_list), "lp_unnorm": sum(unnorm_list), "lp_norm_tokens": norm_list, "lp_unnorm_tokens": unnorm_list}

hf_load_kwargs = {
    "torch_dtype": hf_torch_dtype(),
    "trust_remote_code": False,
}
if HF_DEVICE_MAP_AUTO:
    hf_load_kwargs["device_map"] = "auto"
if HF_ATTN_IMPLEMENTATION is not None:
    hf_load_kwargs["attn_implementation"] = HF_ATTN_IMPLEMENTATION
hf_model = transformers.AutoModelForCausalLM.from_pretrained(
    model_path,
    **hf_load_kwargs,
)
if not HF_DEVICE_MAP_AUTO:
    hf_model = hf_model.to("cuda")
hf_model.eval()
hf_scores = [hf_score(prefix_hf, row) for row in segments_hf]
full_chain_start = extend(segments_hf[0], 8, eos_id)
full_chain_proposals = [extend(row, 8, full_chain_start[0]) for row in segments_hf[1:5]]
hf_replay_scores = {
    tuple(row): hf_score(prefix_hf, row)
    for row in [full_chain_start, *full_chain_proposals]
}
del hf_model
torch.cuda.empty_cache()
gc.collect()

from vllm import LLM, SamplingParams

class ForcedTokenProcessor:
    def __init__(self, target_ids, temperature):
        self.target_ids = [int(x) for x in target_ids]
        self.temperature = float(temperature)
        self.lp_norm = []
        self.lp_unnorm = []

    def clone(self):
        return self

    def __call__(self, prompt_token_ids, past_token_ids, logits):
        pos = len(past_token_ids)
        if pos >= len(self.target_ids):
            return logits
        target = self.target_ids[pos]
        row = logits.float()
        self.lp_norm.append(float(F.log_softmax(row / self.temperature, dim=-1)[target].detach().cpu()))
        self.lp_unnorm.append(float(((1.0 / self.temperature) * F.log_softmax(row, dim=-1)[target]).detach().cpu()))
        forced = torch.full_like(logits, -torch.inf)
        forced[target] = 0.0
        return forced

llm = LLM(
    model=model_path,
    tokenizer=model_path,
    dtype=VLLM_DTYPE,
    trust_remote_code=False,
    tensor_parallel_size=1,
    seed=17,
    enable_prefix_caching=VLLM_ENABLE_PREFIX_CACHING,
    model_impl=VLLM_MODEL_IMPL,
    gpu_memory_utilization=0.35,
    max_model_len=4096,
    enforce_eager=VLLM_ENFORCE_EAGER,
    disable_async_output_proc=VLLM_DISABLE_ASYNC_OUTPUT_PROC,
)

def reset_vllm_prefix_cache_for_scoring():
    if not VLLM_ENABLE_PREFIX_CACHING or not VLLM_RESET_PREFIX_CACHE_FOR_SCORING:
        return
    engine = getattr(llm, "llm_engine", None)
    reset = getattr(engine, "reset_prefix_cache", None)
    if callable(reset):
        reset()

def vllm_score(prefix_ids, target_ids):
    proc = ForcedTokenProcessor(target_ids, TEMP)
    params = SamplingParams(
        max_tokens=len(target_ids),
        min_tokens=0,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        ignore_eos=True,
        skip_special_tokens=False,
        detokenize=False,
        logits_processors=[proc],
    )
    reset_vllm_prefix_cache_for_scoring()
    try:
        outp = llm.generate(prompts=[{"prompt_token_ids": list(prefix_ids)}], sampling_params=params, use_tqdm=False)[0]
    finally:
        reset_vllm_prefix_cache_for_scoring()
    observed = [int(x) for x in outp.outputs[0].token_ids]
    if observed != list(target_ids):
        raise RuntimeError(f"vLLM forced ids mismatch expected={target_ids[:8]} observed={observed[:8]}")
    return {"lp_norm": sum(proc.lp_norm), "lp_unnorm": sum(proc.lp_unnorm), "lp_norm_tokens": list(proc.lp_norm), "lp_unnorm_tokens": list(proc.lp_unnorm)}

vllm_scores = [vllm_score(prefix_hf, row) for row in segments_hf]

score_rows = []
for row_idx, (target_ids, h, v) in enumerate(zip(segments_hf, hf_scores, vllm_scores)):
    seg_norm_diff = abs(h["lp_norm"] - v["lp_norm"])
    seg_unnorm_diff = abs(h["lp_unnorm"] - v["lp_unnorm"])
    seg_tol = ATOL * max(1, len(target_ids))
    seg_passed = seg_norm_diff <= seg_tol and seg_unnorm_diff <= seg_tol
    for pos, token_id in enumerate(target_ids):
        norm_diff = abs(h["lp_norm_tokens"][pos] - v["lp_norm_tokens"][pos])
        unnorm_diff = abs(h["lp_unnorm_tokens"][pos] - v["lp_unnorm_tokens"][pos])
        score_rows.append({
            "row_id": f"score-{row_idx}-{pos}",
            "target_token_id": int(token_id),
            "prefix_len": len(prefix_hf),
            "token_pos": pos,
            "temperature": TEMP,
            "hf_lp_norm": h["lp_norm_tokens"][pos],
            "vllm_lp_norm": v["lp_norm_tokens"][pos],
            "hf_lp_unnorm": h["lp_unnorm_tokens"][pos],
            "vllm_lp_unnorm": v["lp_unnorm_tokens"][pos],
            "norm_abs_diff": norm_diff,
            "unnorm_abs_diff": unnorm_diff,
            "segment_norm_abs_diff": seg_norm_diff,
            "segment_unnorm_abs_diff": seg_unnorm_diff,
            "segment_tolerance": seg_tol,
            "segment_passed": seg_passed,
            "passed": norm_diff <= ATOL and unnorm_diff <= ATOL and seg_passed,
        })

def mh_log_r(cur, prop):
    return prop["lp_unnorm"] + cur["lp_norm"] - cur["lp_unnorm"] - prop["lp_norm"]

mh_rows = []
for i in range(len(segments_hf) - 1):
    cur_h, prop_h = hf_scores[i], hf_scores[i + 1]
    cur_v, prop_v = vllm_scores[i], vllm_scores[i + 1]
    hf_r = mh_log_r(cur_h, prop_h)
    vllm_r = mh_log_r(cur_v, prop_v)
    u = 0.173 + 0.071 * (i % 7)
    log_u = math.log(u)
    suffix_len = max(len(segments_hf[i]), len(segments_hf[i + 1]))
    tol = ATOL * max(1, suffix_len)
    hf_accept = log_u < hf_r
    vllm_accept = log_u < vllm_r
    ambiguous = abs(log_u - hf_r) <= tol
    mh_rows.append({
        "row_id": f"mh-{i}",
        "suffix_len": suffix_len,
        "temperature": TEMP,
        "log_u": log_u,
        "hf_log_r": hf_r,
        "vllm_log_r": vllm_r,
        "abs_diff": abs(hf_r - vllm_r),
        "tolerance": tol,
        "hf_accept": hf_accept,
        "vllm_accept": vllm_accept,
        "ambiguous_boundary": ambiguous,
        "passed": abs(hf_r - vllm_r) <= tol and (hf_accept == vllm_accept or ambiguous),
    })

def chain_replay():
    chain_h = list(full_chain_start)
    chain_v = list(full_chain_start)
    accept_h = []
    accept_v = []
    steps = []
    uniforms = [0.19, 0.41, 0.73, 0.29]
    for i, (proposal, u) in enumerate(zip(full_chain_proposals, uniforms)):
        cur_h = hf_replay_scores[tuple(chain_h)]
        prop_h = hf_replay_scores[tuple(proposal)]
        cur_v = vllm_score(prefix_hf, chain_v)
        prop_v = vllm_score(prefix_hf, proposal)
        hf_r = mh_log_r(cur_h, prop_h)
        vllm_r = mh_log_r(cur_v, prop_v)
        log_u = math.log(u)
        h_acc = log_u < hf_r
        v_acc = log_u < vllm_r
        if h_acc:
            chain_h = list(proposal)
            accept_h.append(i)
        if v_acc:
            chain_v = list(proposal)
            accept_v.append(i)
        steps.append({"step": i, "u": u, "log_u": log_u, "hf_log_r": hf_r, "vllm_log_r": vllm_r, "hf_accept": h_acc, "vllm_accept": v_acc, "log_r_abs_diff": abs(hf_r - vllm_r)})
    return {
        "passed": chain_h == chain_v and accept_h == accept_v,
        "final_token_chain_match": chain_h == chain_v,
        "acceptance_count_match": len(accept_h) == len(accept_v),
        "acceptance_positions_match": accept_h == accept_v,
        "hf_accept_positions": accept_h,
        "vllm_accept_positions": accept_v,
        "hf_final_token_ids": chain_h,
        "vllm_final_token_ids": chain_v,
        "steps": steps,
        "tokenizer_parity": tokenizer_parity,
        "runtime_metadata": {
            "backend": "vllm",
            "vllm_version": __import__("importlib.metadata").metadata.version("vllm"),
            "vllm_scoring_mode": "forced_decode_v0",
            "VLLM_ATTENTION_BACKEND": os.environ.get("VLLM_ATTENTION_BACKEND"),
            "dtype": VLLM_DTYPE,
            "model_impl": VLLM_MODEL_IMPL,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "hf_dtype": HF_DTYPE,
            "hf_device_map_auto": HF_DEVICE_MAP_AUTO,
            "hf_attn_implementation": HF_ATTN_IMPLEMENTATION,
            "hf_scoring_mode": HF_SCORING_MODE,
            "vllm_enforce_eager": VLLM_ENFORCE_EAGER,
            "vllm_disable_async_output_proc": VLLM_DISABLE_ASYNC_OUTPUT_PROC,
            "vllm_enable_prefix_caching": VLLM_ENABLE_PREFIX_CACHING,
            "vllm_reset_prefix_cache_for_scoring": VLLM_RESET_PREFIX_CACHE_FOR_SCORING,
        },
    }

full_chain = chain_replay()
score_diffs = [max(r["norm_abs_diff"], r["unnorm_abs_diff"]) for r in score_rows]
ambig = sum(1 for r in mh_rows if r["ambiguous_boundary"])
score_summary = {
    "kind": "score_parity",
    "tolerance": ATOL,
    "n_rows": len(score_rows),
    "passed": bool(score_rows) and all(r["passed"] for r in score_rows),
    "max_abs_diff": max(score_diffs) if score_diffs else None,
    "mean_abs_diff": statistics.mean(score_diffs) if score_diffs else None,
    "max_segment_abs_diff": max(max(r["segment_norm_abs_diff"], r["segment_unnorm_abs_diff"]) for r in score_rows),
}
mh_summary = {
    "kind": "mh_replay_parity",
    "n_rows": len(mh_rows),
    "passed": bool(mh_rows) and all(r["passed"] for r in mh_rows) and (ambig / len(mh_rows)) <= AMBIG_LIMIT,
    "max_abs_diff": max((r["abs_diff"] for r in mh_rows), default=None),
    "ambiguous_boundary_count": ambig,
    "ambiguous_boundary_rate": ambig / len(mh_rows) if mh_rows else 0.0,
    "ambiguous_boundary_rate_limit": AMBIG_LIMIT,
}
summary = {
    "passed": score_summary["passed"] and mh_summary["passed"] and full_chain["passed"],
    "hf_oracle": {
        "dtype": HF_DTYPE,
        "device_map_auto": HF_DEVICE_MAP_AUTO,
        "attn_implementation": HF_ATTN_IMPLEMENTATION,
        "scoring_mode": HF_SCORING_MODE,
    },
    "vllm_candidate": {
        "dtype": VLLM_DTYPE,
        "model_impl": VLLM_MODEL_IMPL,
        "scoring_mode": "forced_decode_v0",
        "VLLM_ATTENTION_BACKEND": os.environ.get("VLLM_ATTENTION_BACKEND"),
        "enforce_eager": VLLM_ENFORCE_EAGER,
        "disable_async_output_proc": VLLM_DISABLE_ASYNC_OUTPUT_PROC,
        "enable_prefix_caching": VLLM_ENABLE_PREFIX_CACHING,
        "reset_prefix_cache_for_scoring": VLLM_RESET_PREFIX_CACHE_FOR_SCORING,
    },
    "score_parity": score_summary,
    "mh_replay_parity": mh_summary,
    "full_chain_replay": full_chain,
    "artifacts": {
        "score_parity": "score_parity.jsonl",
        "mh_replay_parity": "mh_replay_parity.jsonl",
        "full_chain_replay": "full_chain_replay.json",
        "summary": "calibration_summary.json",
        "report": "calibration_report.md",
    },
}

def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))

write_jsonl(cal / "score_parity.jsonl", score_rows)
write_jsonl(cal / "mh_replay_parity.jsonl", mh_rows)
write_json(cal / "full_chain_replay.json", full_chain)
write_json(cal / "calibration_summary.json", summary)
(cal / "calibration_report.md").write_text(
    "# vLLM/HF calibration\n\n"
    f"- passed: {summary['passed']}\n"
    f"- score_rows: {score_summary['n_rows']}\n"
    f"- score_max_abs_diff: {score_summary['max_abs_diff']}\n"
    f"- score_mean_abs_diff: {score_summary['mean_abs_diff']}\n"
    f"- score_max_segment_abs_diff: {score_summary['max_segment_abs_diff']}\n"
    f"- mh_rows: {mh_summary['n_rows']}\n"
    f"- mh_max_abs_diff: {mh_summary['max_abs_diff']}\n"
    f"- mh_ambiguous_boundary_rate: {mh_summary['ambiguous_boundary_rate']}\n"
)
write_json(out / "calibration_summary.json", summary)
(out / "calibration_report.md").write_text((cal / "calibration_report.md").read_text())
print("POLARIS_FLOW_VLLM_HF_CALIBRATION_SUMMARY_BEGIN")
print(json.dumps(summary, indent=2, sort_keys=True))
print("POLARIS_FLOW_VLLM_HF_CALIBRATION_SUMMARY_END")
if EMIT_TGZ:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for path in sorted(out.rglob("*")):
            if path.is_file():
                tar.add(path, arcname=str(path.relative_to(out)))
    print("POLARIS_FLOW_VLLM_HF_CALIBRATION_TGZ_B64_BEGIN")
    print(base64.b64encode(buf.getvalue()).decode("ascii"))
    print("POLARIS_FLOW_VLLM_HF_CALIBRATION_TGZ_B64_END")
if not summary["passed"]:
    raise SystemExit(1)
print(f"POLARIS_FLOW_VLLM_HF_CALIBRATION_OK {out}")
