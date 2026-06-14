from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else ROOT / "config.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = str(ROOT)
    cfg["_config_path"] = str(cfg_path)

    profile_name = cfg.get("active_profile", "local")
    if profile_name not in cfg.get("profiles", {}):
        raise ValueError(
            f"config.yaml 中 active_profile={profile_name!r} 不存在，"
            f"可选: {list(cfg['profiles'].keys())}"
        )
    profile = cfg["profiles"][profile_name]
    cfg["active"] = profile
    cfg["active_profile_name"] = profile_name
    return cfg


def project_path(cfg: dict[str, Any], key: str) -> Path:
    p = ROOT / cfg["project"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p


def data_dir(cfg: dict[str, Any]) -> Path:
    return project_path(cfg, "data_dir")


def output_dir(cfg: dict[str, Any]) -> Path:
    return project_path(cfg, "output_dir")


def results_dir(cfg: dict[str, Any]) -> Path:
    return project_path(cfg, "results_dir")


def resolve_model_path(model_name: str) -> str:
    """本地目录原样返回；HuggingFace 模型 ID 原样返回。"""
    if not model_name:
        raise ValueError("model.name 未配置")
    path = Path(model_name)
    if path.exists() or (not model_name.startswith(("Qwen/", "meta-llama/", "mistralai/")) and path.is_absolute()):
        return str(path.resolve())
    return model_name


def validate_local_model(model_name: str) -> None:
    """本地路径必须包含 config.json，避免误触发在线下载。"""
    path = Path(model_name)
    if not path.exists():
        return
    config_file = path / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(
            f"本地模型目录不完整: {path}\n"
            f"缺少 config.json。请确认 scp 上传完整，或改用 HuggingFace 模型 ID。"
        )


def get_num_train_epochs(tcfg: dict[str, Any], train_size: int) -> float:
    by_size = tcfg.get("epochs_by_size") or {}
    if train_size in by_size:
        return float(by_size[train_size])
    if str(train_size) in by_size:
        return float(by_size[str(train_size)])
    return float(tcfg["num_train_epochs"])


def print_run_context(cfg: dict[str, Any], *, script: str, extra: str | None = None) -> None:
    active = cfg["active"]
    mcfg = active["model"]
    model_name = mcfg["name"]
    print("=" * 60)
    print(f"[{script}] 运行配置检查")
    print(f"  config     : {cfg.get('_config_path', ROOT / 'config.yaml')}")
    print(f"  profile    : {cfg['active_profile_name']}")
    print(f"  model      : {model_name}")
    print(f"  use_4bit   : {mcfg.get('use_4bit', True)}")
    print(f"  train_sizes: {active['data'].get('train_sizes')}")
    print(f"  holdout    : {active['data'].get('holdout_size')}")
    if extra:
        print(f"  {extra}")
    print("=" * 60)

    path = Path(model_name)
    if path.exists():
        validate_local_model(model_name)
        print(f"[OK] 本地模型目录: {path.resolve()}")
    elif "/" in model_name and not model_name.startswith(("/", "\\")):
        print(f"[WARN] 将从 HuggingFace 加载: {model_name}")
        print("       若已 scp 上传模型，请将 config.yaml 的 model.name 改为本地绝对路径。")
    print()


def ensure_train_files(cfg: dict[str, Any], train_sizes: list[int]) -> None:
    missing = []
    ddir = data_dir(cfg)
    for size in train_sizes:
        path = ddir / f"train_{size}.jsonl"
        if not path.exists():
            missing.append(str(path))
    if missing:
        sizes = " ".join(str(s) for s in train_sizes)
        raise FileNotFoundError(
            "缺少训练数据文件:\n  " + "\n  ".join(missing) + "\n"
            f"请运行: python scripts/prepare_data.py --all_sizes\n"
            f"或分别运行: python scripts/prepare_data.py --train_size <size>"
        )

    holdout_size = cfg["active"]["data"]["holdout_size"]
    holdout_path = ddir / f"holdout_{holdout_size}.jsonl"
    if not holdout_path.exists():
        raise FileNotFoundError(
            f"缺少 holdout 文件: {holdout_path}\n"
            f"请运行: python scripts/prepare_data.py --all_sizes"
        )
