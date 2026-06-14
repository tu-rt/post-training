"""
准备 COIG-CQIA 训练集与 hold-out。

用法:
  python scripts/prepare_data.py
  python scripts/prepare_data.py --train_size 200 --holdout_size 20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import data_dir, load_config, print_run_context
from src.data_utils import load_coig_cqia, save_jsonl, split_train_holdout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--train_size", type=int, default=None)
    parser.add_argument("--holdout_size", type=int, default=None)
    parser.add_argument(
        "--all_sizes",
        action="store_true",
        help="按 config 中 train_sizes 批量生成所有 train_*.jsonl",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    print_run_context(cfg, script="prepare_data")
    active = cfg["active"]
    dcfg = cfg["dataset"]
    data_cfg = active["data"]

    holdout_size = args.holdout_size or data_cfg["holdout_size"]
    if args.all_sizes:
        train_sizes = sorted(set(data_cfg["train_sizes"]))
    else:
        train_sizes = [args.train_size or max(data_cfg["train_sizes"])]

    cache = data_dir(cfg) / "coig_cqia_clean.json"
    print(f"加载 {dcfg['sft_source']} …")
    rows = load_coig_cqia(
        source=dcfg["sft_source"],
        seed=dcfg["seed"],
        min_chars=dcfg["min_chars"],
        max_chars=dcfg["max_chars"],
        subset=dcfg.get("sft_subset"),
        cache_path=cache,
    )
    print(f"清洗后共 {len(rows)} 条，缓存: {cache}")

    ddir = data_dir(cfg)
    max_size = max(train_sizes)
    _, holdout = split_train_holdout(
        rows, train_size=max_size, holdout_size=holdout_size, seed=dcfg["seed"]
    )
    out_hold = ddir / f"holdout_{holdout_size}.jsonl"
    save_jsonl(holdout, out_hold)
    print(f"Hold-out: {out_hold} ({len(holdout)} 条)")

    generated = []
    for train_size in train_sizes:
        train, _ = split_train_holdout(
            rows, train_size=train_size, holdout_size=holdout_size, seed=dcfg["seed"]
        )
        out_train = ddir / f"train_{train_size}.jsonl"
        save_jsonl(train, out_train)
        generated.append({"train_size": train_size, "path": str(out_train), "n": len(train)})
        print(f"训练集: {out_train} ({len(train)} 条)")

    meta = {
        "source": dcfg["sft_source"],
        "train_sizes": train_sizes,
        "holdout_size": holdout_size,
        "holdout_path": str(out_hold),
        "train_files": generated,
        "profile": cfg["active_profile_name"],
    }
    meta_path = ddir / "data_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"元信息: {meta_path}")


if __name__ == "__main__":
    main()
