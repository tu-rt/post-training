"""
恢复流水线：重训崩溃的 E2/E3（1 epoch）→ 评测 → 汇总。

适用场景：
  - E2/E3 holdout 出现大量重复「!」（模式崩溃）
  - run_overnight.py 在 E2 检查处中止

用法:
  nohup python scripts/run_recover.py >> logs/recover_stdout.log 2>&1 &
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config, output_dir, results_dir  # noqa: E402

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "recover_latest.log"
REPORT_FILE = LOG_DIR / "recover_report.txt"

_SUB_ENV = {
    **os.environ,
    "PYTHONUTF8": "1",
    "NCCL_P2P_DISABLE": "1",
    "NCCL_IB_DISABLE": "1",
}
_INFER_ENV = {**_SUB_ENV, "CUDA_VISIBLE_DEVICES": "0"}

EXPERIMENTS = [
    {"exp_id": "E2", "train_size": 5000},
    {"exp_id": "E3", "train_size": 10000},
]


def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def lr_tag(lr: float) -> str:
    return f"{lr:.0e}".replace("e-0", "e-").replace("e+", "e")


def run_dir(exp_id: str, train_size: int, lr: float) -> Path:
    return output_dir(load_config()) / f"{exp_id}_size{train_size}_lr{lr_tag(lr)}"


def adapter_dir(exp_id: str, train_size: int, lr: float) -> Path:
    return run_dir(exp_id, train_size, lr) / "adapter"


def run_step(name: str, cmd: list[str], *, infer_gpu: bool = False) -> None:
    log(f"===== START: {name} =====")
    log("CMD: " + " ".join(cmd))
    env = _INFER_ENV if infer_gpu else _SUB_ENV
    subprocess.check_call(cmd, cwd=str(ROOT), env=env)
    log(f"===== DONE: {name} =====\n")


def holdout_bang_ratio(exp_id: str) -> tuple[float, float]:
    path = results_dir(load_config()) / f"holdout_{exp_id}_detail.csv"
    df = pd.read_csv(path)
    bang = df["lora_answer"].astype(str).str.fullmatch(r"!+").sum()
    return bang / len(df), float(df["lora_score"].mean())


def sanity_check_adapter(adapter_path: Path) -> str:
    import json as _json

    from src.model_utils import generate_answer, load_model_for_inference

    cfg = load_config()
    mcfg = cfg["active"]["model"]
    model, tok = load_model_for_inference(
        mcfg["name"],
        adapter_path=str(adapter_path),
        use_4bit=mcfg.get("use_4bit", True),
    )
    holdout = ROOT / "data" / f"holdout_{cfg['active']['data']['holdout_size']}.jsonl"
    with open(holdout, encoding="utf-8") as f:
        item = _json.loads(f.readline())
    preview = generate_answer(model, tok, item["instruction"], max_new_tokens=64)
    del model, tok
    if "!" * 20 in preview:
        raise RuntimeError(f"生成崩溃（重复!）: {preview[:80]!r}")
    return preview


def backup_run(exp_id: str, train_size: int, lr: float, suffix: str = "broken_2epoch") -> None:
    src = run_dir(exp_id, train_size, lr)
    if not src.exists():
        return
    dst = output_dir(load_config()) / f"{exp_id}_size{train_size}_lr{lr_tag(lr)}_{suffix}"
    if dst.exists():
        log(f"备份已存在: {dst}")
        if src.exists():
            shutil.rmtree(src)
            log(f"删除未完成的训练目录: {src}")
        return
    shutil.move(str(src), str(dst))
    log(f"已备份 {src.name} → {dst.name}")


def train_eval_exp(exp_id: str, train_size: int, lr: float) -> Path:
    py = sys.executable
    run_step(
        f"{exp_id} 训练 size={train_size} lr={lr}",
        [
            py,
            "scripts/train_lora.py",
            "--train_size",
            str(train_size),
            "--exp_id",
            exp_id,
            "--learning_rate",
            str(lr),
        ],
    )
    adapter = adapter_dir(exp_id, train_size, lr)
    preview = sanity_check_adapter(adapter)
    log(f"{exp_id} 抽查 OK: {preview[:100]}")

    run_step(
        f"{exp_id} C-Eval",
        [
            py,
            "scripts/evaluate_ceval.py",
            "--mode",
            "lora",
            "--adapter",
            str(adapter),
            "--exp_id",
            exp_id,
        ],
        infer_gpu=True,
    )
    run_step(
        f"{exp_id} Hold-out",
        [
            py,
            "scripts/evaluate_holdout.py",
            "--adapter",
            str(adapter),
            "--exp_id",
            exp_id,
        ],
        infer_gpu=True,
    )
    bang, score = holdout_bang_ratio(exp_id)
    log(f"{exp_id} holdout: bang={bang:.1%}, lora_score={score:.4f}")
    if bang > 0.1:
        raise RuntimeError(f"{exp_id} holdout 仍异常（{bang:.0%} 为纯!）")
    return adapter


def recover_exp(exp_id: str, train_size: int) -> dict:
    backup_run(exp_id, train_size, 2e-4)
    try:
        adapter = train_eval_exp(exp_id, train_size, lr=2e-4)
        return {"status": "ok", "lr": 2e-4, "adapter": str(adapter)}
    except Exception as exc:
        log(f"{exp_id} lr=2e-4 失败: {exc}，尝试 lr=1e-4")
        backup_run(exp_id, train_size, 2e-4, suffix="broken_retry")
        adapter = train_eval_exp(exp_id, train_size, lr=1e-4)
        return {"status": "ok_retry", "lr": 1e-4, "adapter": str(adapter), "first_error": str(exc)}


def write_ablation_summary() -> None:
    cfg = load_config()
    res = results_dir(cfg)
    train_sizes = cfg["active"]["data"]["train_sizes"]
    rows = []
    if (res / "ceval_E0_meta.json").exists():
        with open(res / "ceval_E0_meta.json", encoding="utf-8") as f:
            e0 = json.load(f)
        rows.append({"exp_id": "E0", "train_size": 0, "ceval_acc": e0["accuracy"]})
    for i, size in enumerate(train_sizes, start=1):
        exp_id = f"E{i}"
        row: dict = {"exp_id": exp_id, "train_size": size}
        meta = res / f"ceval_{exp_id}_meta.json"
        if meta.exists():
            with open(meta, encoding="utf-8") as f:
                row["ceval_acc"] = json.load(f)["accuracy"]
        hold = res / f"holdout_{exp_id}_summary.csv"
        if hold.exists():
            h = pd.read_csv(hold)
            row["holdout_win_rate"] = float(h["lora_win_rate"].iloc[0])
            row["holdout_lora_score"] = float(h["lora_score_mean"].iloc[0])
        rows.append(row)
    summary = pd.DataFrame(rows)
    out = res / "ablation_summary.csv"
    summary.to_csv(out, index=False, encoding="utf-8-sig")
    log("=== Ablation 汇总 ===\n" + summary.to_string(index=False))


def main() -> None:
    if LOG_FILE.exists():
        LOG_FILE.unlink()
    log("恢复流水线启动")

    status: dict = {"steps": {}, "success": False}
    try:
        run_step("预检", [sys.executable, "scripts/preflight_check.py"])
        for exp in EXPERIMENTS:
            result = recover_exp(exp["exp_id"], exp["train_size"])
            status["steps"][exp["exp_id"]] = result
        write_ablation_summary()
        status["success"] = True
        log("恢复流水线完成 ✓")
    except Exception as exc:
        log(f"恢复失败: {exc}\n{traceback.format_exc()}")
        status["error"] = str(exc)
    finally:
        REPORT_FILE.write_text(
            f"恢复流水线报告 {datetime.now()}\n{json.dumps(status, ensure_ascii=False, indent=2)}\n",
            encoding="utf-8",
        )
        if not status["success"]:
            sys.exit(1)


if __name__ == "__main__":
    main()
