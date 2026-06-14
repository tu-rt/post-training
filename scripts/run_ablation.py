"""
按 config 中的 train_sizes 批量训练 + 评测，汇总 ablation 表。

用法（本机 local profile）:
  python scripts/run_ablation.py --steps prepare,train,eval
  python scripts/run_ablation.py --steps eval --skip_e0
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import ensure_train_files, load_config, output_dir, print_run_context, results_dir

# 确保子进程继承环境变量 + UTF-8 模式（修复 Windows GBK 问题）
_SUB_ENV = {**os.environ}
_SUB_ENV["PYTHONUTF8"] = "1"
# 多卡 NCCL P2P 不可用时禁用，避免训练通信错误（问题 #10）
_SUB_ENV.setdefault("NCCL_P2P_DISABLE", "1")
_SUB_ENV.setdefault("NCCL_IB_DISABLE", "1")


def run_cmd(cmd: list[str]) -> None:
    print("\n>>>", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(ROOT), env=_SUB_ENV)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument(
        "--steps",
        default="prepare,train,eval",
        help="prepare,train,eval 逗号分隔",
    )
    parser.add_argument("--skip_e0", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    active = cfg["active"]
    data_cfg = active["data"]
    train_sizes = data_cfg["train_sizes"]
    holdout = data_cfg["holdout_size"]
    lr = active["train"]["learning_rate"]
    lr_tag = f"{lr:.0e}".replace("e-0", "e-").replace("e+", "e")
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    py = sys.executable

    print_run_context(cfg, script="run_ablation", extra=f"steps: {','.join(steps)}")

    if "prepare" in steps:
        run_cmd([py, "scripts/prepare_data.py", "--all_sizes", "--holdout_size", str(holdout)])

    if "train" in steps or "eval" in steps:
        ensure_train_files(cfg, train_sizes)

    adapters: dict[str, str] = {}

    if "train" in steps:
        for i, size in enumerate(train_sizes, start=1):
            exp_id = f"E{i}"
            run_cmd(
                [
                    py,
                    "scripts/train_lora.py",
                    "--train_size",
                    str(size),
                    "--exp_id",
                    exp_id,
                ]
            )
            adapter = output_dir(cfg) / f"{exp_id}_size{size}_lr{lr_tag}" / "adapter"
            adapters[exp_id] = str(adapter)

    if "eval" in steps:
        if not args.skip_e0:
            run_cmd([py, "scripts/evaluate_ceval.py", "--mode", "base", "--exp_id", "E0"])

        for i, size in enumerate(train_sizes, start=1):
            exp_id = f"E{i}"
            if exp_id not in adapters:
                adapters[exp_id] = str(
                    output_dir(cfg) / f"{exp_id}_size{size}_lr{lr_tag}" / "adapter"
                )
            run_cmd(
                [
                    py,
                    "scripts/evaluate_ceval.py",
                    "--mode",
                    "lora",
                    "--adapter",
                    adapters[exp_id],
                    "--exp_id",
                    exp_id,
                ]
            )
            run_cmd(
                [
                    py,
                    "scripts/evaluate_holdout.py",
                    "--adapter",
                    adapters[exp_id],
                    "--exp_id",
                    exp_id,
                ]
            )

    # 汇总
    res = results_dir(cfg)
    rows = []
    if (res / "ceval_E0_meta.json").exists():
        with open(res / "ceval_E0_meta.json", encoding="utf-8") as f:
            e0 = json.load(f)
        rows.append({"exp_id": "E0", "train_size": 0, "ceval_acc": e0["accuracy"]})

    for i, size in enumerate(train_sizes, start=1):
        exp_id = f"E{i}"
        row = {"exp_id": exp_id, "train_size": size}
        meta_path = res / f"ceval_{exp_id}_meta.json"
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                row["ceval_acc"] = json.load(f)["accuracy"]
        hold_path = res / f"holdout_{exp_id}_summary.csv"
        if hold_path.exists():
            h = pd.read_csv(hold_path)
            row["holdout_win_rate"] = float(h["lora_win_rate"].iloc[0])
            row["holdout_lora_score"] = float(h["lora_score_mean"].iloc[0])
        rows.append(row)

    summary = pd.DataFrame(rows)
    out_path = res / "ablation_summary.csv"
    summary.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("\n=== Ablation 汇总 ===")
    print(summary.to_string(index=False))
    print(f"\n已保存: {out_path}")


if __name__ == "__main__":
    main()
