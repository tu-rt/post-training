"""
一键跑通本机冒烟：准备小数据 → 训练 30 step → C-Eval + Hold-out

用法:
  python scripts/smoke_test.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 继承父进程环境变量（确保 HF_ENDPOINT 等能传到子进程）
_SUB_ENV = {**os.environ}
_SUB_ENV["PYTHONUTF8"] = "1"
_SUB_ENV.setdefault("NCCL_P2P_DISABLE", "1")
_SUB_ENV.setdefault("NCCL_IB_DISABLE", "1")


def run(cmd: list[str]) -> None:
    print("\n>>>", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(ROOT), env=_SUB_ENV)


def main() -> None:
    py = sys.executable
    train_size = 200
    holdout = 20

    run([py, "scripts/prepare_data.py", "--train_size", str(train_size), "--holdout_size", str(holdout)])
    run(
        [
            py,
            "scripts/train_lora.py",
            "--train_size",
            str(train_size),
            "--exp_id",
            "E1_smoke",
            "--max_steps",
            "30",
        ]
    )

    adapter = ROOT / "outputs" / f"E1_smoke_size{train_size}_lr2e-4" / "adapter"
    if not adapter.exists():
        candidates = sorted(ROOT.glob("outputs/E1_smoke_*"))
        if candidates:
            adapter = candidates[-1] / "adapter"
    if not adapter.exists():
        raise FileNotFoundError("未找到 smoke 训练产物，请检查 outputs/ 目录")

    run([py, "scripts/evaluate_ceval.py", "--mode", "base", "--exp_id", "E0_smoke"])
    run(
        [
            py,
            "scripts/evaluate_ceval.py",
            "--mode",
            "lora",
            "--adapter",
            str(adapter),
            "--exp_id",
            "E1_smoke",
        ]
    )
    holdout_file = ROOT / "data" / f"holdout_{holdout}.jsonl"
    run(
        [
            py,
            "scripts/evaluate_holdout.py",
            "--adapter",
            str(adapter),
            "--exp_id",
            "E1_smoke",
            "--holdout_file",
            str(holdout_file),
            "--max_samples",
            "5",
        ]
    )
    print("\n冒烟完成。查看 results/ 目录。")


if __name__ == "__main__":
    main()
