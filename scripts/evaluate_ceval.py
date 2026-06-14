"""
C-Eval 子集评测（base 或 LoRA adapter）。

用法:
  python scripts/evaluate_ceval.py --mode base
  python scripts/evaluate_ceval.py --mode lora --adapter outputs/E1_size200_lr0.0002/adapter
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
from src.eval_ceval import load_ceval_subset, summarize_ceval
from src.model_utils import extract_choice_letter, free_gpu, generate_answer, load_model_for_inference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--mode", choices=["base", "lora"], default="base")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--exp_id", default="E0")
    parser.add_argument("--max_new_tokens", type=int, default=16)
    args = parser.parse_args()

    cfg = load_config(args.config)
    print_run_context(cfg, script="evaluate_ceval", extra=f"mode={args.mode}, exp_id={args.exp_id}")
    active = cfg["active"]
    mcfg = active["model"]
    ccfg = cfg["ceval"]
    data_cfg = active["data"]

    cache = data_dir(cfg) / "ceval_subset.json"
    samples = load_ceval_subset(
        dataset_name=ccfg["name"],
        subjects=ccfg["subjects"],
        per_subject=data_cfg["ceval_per_subject"],
        seed=cfg["dataset"]["seed"],
        cache_path=cache,
    )
    print(f"C-Eval 子集: {len(samples)} 题")

    model_name = mcfg["name"]
    adapter = args.adapter if args.mode == "lora" else None
    if args.mode == "lora" and not adapter:
        raise ValueError("lora 模式需指定 --adapter 路径")
    if adapter:
        adapter_path = Path(adapter)
        if not adapter_path.is_absolute():
            adapter_path = ROOT / adapter_path
        if not adapter_path.exists():
            raise FileNotFoundError(f"找不到 adapter: {adapter_path}")
        check_adapter_weights(adapter_path)
        adapter = str(adapter_path)

    model, tokenizer = load_model_for_inference(
        model_name, adapter_path=adapter, use_4bit=mcfg.get("use_4bit", True)
    )

    records = []
    max_input = mcfg.get("max_seq_length", 2048)
    for item in tqdm(samples, desc=f"eval-{args.exp_id}"):
        pred_text = generate_answer(
            model,
            tokenizer,
            item["prompt"],
            max_new_tokens=args.max_new_tokens,
            max_input_tokens=max_input,
        )
        pred = extract_choice_letter(pred_text) or "?"
        records.append(
            {
                "id": item["id"],
                "subject": item["subject"],
                "gold": item["gold"],
                "pred": pred,
                "correct": int(pred == item["gold"]),
                "raw": pred_text[:80],
            }
        )

    del model, tokenizer
    free_gpu()
    df = pd.DataFrame(records)
    summary = summarize_ceval(records)

    out = results_dir(cfg)
    tag = args.exp_id
    df.to_csv(out / f"ceval_{tag}_detail.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / f"ceval_{tag}_summary.csv", index=False, encoding="utf-8-sig")

    with open(out / f"ceval_{tag}_meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "exp_id": tag,
                "mode": args.mode,
                "adapter": adapter,
                "base_model": model_name,
                "n": len(records),
                "accuracy": float(df["correct"].mean()),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n=== C-Eval 汇总 ===")
    print(summary.to_string(index=False))
    print(f"\naccuracy = {df['correct'].mean():.4f}")


if __name__ == "__main__":
    main()
