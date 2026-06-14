"""
启动前环境/配置自检（对应异常汇总第四节检查清单）。

用法:
  python scripts/preflight_check.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import ensure_train_files, load_config, print_run_context, resolve_model_path, validate_local_model


def main() -> None:
    cfg = load_config()
    print_run_context(cfg, script="preflight_check")

    try:
        import torch

        print(f"PyTorch     : {torch.__version__}")
        print(f"CUDA 可用   : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA 版本   : {torch.version.cuda}")
            print(f"GPU 数量    : {torch.cuda.device_count()}")
    except Exception as exc:
        print(f"[FAIL] PyTorch 检查失败: {exc}")
        sys.exit(1)

    try:
        import transformers
        import trl
        import peft

        print(f"transformers: {transformers.__version__}")
        print(f"trl         : {trl.__version__}")
        print(f"peft        : {peft.__version__}")
    except Exception as exc:
        print(f"[FAIL] 依赖导入失败: {exc}")
        sys.exit(1)

    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"[WARN] pip check 有冲突:\n{result.stdout}{result.stderr}")
    else:
        print("[OK] pip check 通过")

    model_name = resolve_model_path(cfg["active"]["model"]["name"])
    path = Path(model_name)
    if path.exists():
        validate_local_model(model_name)
        print(f"[OK] 本地模型: {path}")
    else:
        print(f"[INFO] 模型将从 HuggingFace 加载: {model_name}")

    train_sizes = cfg["active"]["data"]["train_sizes"]
    try:
        ensure_train_files(cfg, train_sizes)
        print(f"[OK] 训练/holdout 数据文件齐全: {train_sizes}")
    except FileNotFoundError as exc:
        print(f"[WARN] {exc}")

    print("\n预检完成。若 profile/model/数据均 OK，可运行 run_ablation 或单独脚本。")


if __name__ == "__main__":
    main()
