"""
Hold-out 集上对比 base vs LoRA（含简单 win-rate）。

用法:
  python scripts/evaluate_holdout.py --adapter outputs/E1_size200_lr0.0002/adapter --exp_id E1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.adapter_utils import check_adapter_weights
from src.config import data_dir, load_config, print_run_context, results_dir
from src.model_utils import free_gpu, generate_answer, load_model_for_inference


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize(s: str) -> str:
    return "".join(str(s).lower().split())


def overlap_score(pred: str, ref: str) -> float:
    p, r = normalize(pred), normalize(ref)
    if not p or not r:
        return 0.0
    if r in p or p in r:
        return 1.0
    ps, rs = set(p), set(r)
    inter = len(ps & rs)
    return inter / max(len(rs), 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--exp_id", default="E1")
    parser.add_argument("--holdout_file", default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    print_run_context(cfg, script="evaluate_holdout", extra=f"exp_id={args.exp_id}")
    active = cfg["active"]
    mcfg = active["model"]
    holdout_size = active["data"]["holdout_size"]

    holdout_path = (
        Path(args.holdout_file)
        if args.holdout_file
        else data_dir(cfg) / f"holdout_{holdout_size}.jsonl"
    )
    if not holdout_path.exists():
        raise FileNotFoundError(f"找不到 {holdout_path}，请先 prepare_data")

    rows = load_jsonl(holdout_path)
    if args.max_samples:
        rows = rows[: args.max_samples]

    model_name = mcfg["name"]
    base_model, base_tok = load_model_for_inference(
        model_name, adapter_path=None, use_4bit=mcfg.get("use_4bit", True)
    )

    max_input = mcfg.get("max_seq_length", 2048)
    records = []
    for i, item in enumerate(tqdm(rows, desc="holdout-base")):
        prompt = item["instruction"]
        ref = item["response"]
        base_ans = generate_answer(
            base_model,
            base_tok,
            prompt,
            max_new_tokens=256,
            max_input_tokens=max_input,
        )
        records.append(
            {
                "id": i,
                "instruction": prompt,
                "reference": ref,
                "base_answer": base_ans,
                "lora_answer": "",
                "base_score": overlap_score(base_ans, ref),
                "lora_score": 0.0,
            }
        )
    del base_model, base_tok
    free_gpu()

    adapter_path = Path(args.adapter)
    if not adapter_path.is_absolute():
        adapter_path = ROOT / adapter_path
    if not adapter_path.exists():
        raise FileNotFoundError(f"找不到 adapter: {adapter_path}")
    check_adapter_weights(adapter_path)

    lora_model, lora_tok = load_model_for_inference(
        model_name, adapter_path=str(adapter_path), use_4bit=mcfg.get("use_4bit", True)
    )
    for i, item in enumerate(tqdm(rows, desc="holdout-lora")):
        lora_ans = generate_answer(
            lora_model,
            lora_tok,
            item["instruction"],
            max_new_tokens=256,
            max_input_tokens=max_input,
        )
        records[i]["lora_answer"] = lora_ans
        records[i]["lora_score"] = overlap_score(lora_ans, item["response"])
        records[i]["lora_win"] = int(records[i]["lora_score"] > records[i]["base_score"])
    del lora_model, lora_tok
    free_gpu()

    df = pd.DataFrame(records)
    win_rate = df["lora_win"].mean()
    base_mean = df["base_score"].mean()
    lora_mean = df["lora_score"].mean()

    out = results_dir(cfg)
    tag = args.exp_id
    df.to_csv(out / f"holdout_{tag}_detail.csv", index=False, encoding="utf-8-sig")
    summary = pd.DataFrame(
        [
            {
                "exp_id": tag,
                "n": len(df),
                "base_score_mean": round(base_mean, 4),
                "lora_score_mean": round(lora_mean, 4),
                "lora_win_rate": round(win_rate, 4),
            }
        ]
    )
    summary.to_csv(out / f"holdout_{tag}_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
