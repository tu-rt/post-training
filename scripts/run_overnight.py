"""
夜间无人值守流水线（E2 评测 → E3 重训 → E3 评测 → 汇总）。

用法（在项目根目录）:
  mkdir -p logs
  nohup python scripts/run_overnight.py >> logs/overnight_stdout.log 2>&1 &

  查看进度:
    tail -f logs/overnight_stdout.log
    tail -f logs/overnight_latest.log

  第二天早上看结果:
    cat logs/overnight_report.txt
    cat results/ablation_summary.csv
"""
from __future__ import annotations

import json
import os
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
LOG_FILE = LOG_DIR / "overnight_latest.log"
REPORT_FILE = LOG_DIR / "overnight_report.txt"

_SUB_ENV = {**os.environ, "PYTHONUTF8": "1", "NCCL_P2P_DISABLE": "1"}
_INFER_ENV = {**_SUB_ENV, "CUDA_VISIBLE_DEVICES": "0"}


def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def lr_tag(lr: float) -> str:
    return f"{lr:.0e}".replace("e-0", "e-").replace("e+", "e")


def adapter_dir(exp_id: str, train_size: int, lr: float) -> Path:
    return output_dir(load_config()) / f"{exp_id}_size{train_size}_lr{lr_tag(lr)}" / "adapter"


def run_step(name: str, cmd: list[str], *, infer_gpu: bool = False) -> None:
    log(f"===== START: {name} =====")
    log("CMD: " + " ".join(cmd))
    env = _INFER_ENV if infer_gpu else _SUB_ENV
    subprocess.check_call(cmd, cwd=str(ROOT), env=env)
    log(f"===== DONE: {name} =====\n")


def holdout_bang_ratio(exp_id: str) -> tuple[float, float]:
    path = results_dir(load_config()) / f"holdout_{exp_id}_detail.csv"
    if not path.exists():
        return -1.0, -1.0
    df = pd.read_csv(path)
    bang = df["lora_answer"].astype(str).str.fullmatch(r"!+").sum()
    return bang / len(df), float(df["lora_score"].mean())


def sanity_check_adapter(adapter_path: Path, max_new_tokens: int = 64) -> str:
    """返回模型生成预览；若检测到重复 ! 则抛异常。"""
    import json as _json

    from src.config import load_config as _load_config
    from src.model_utils import generate_answer, load_model_for_inference

    cfg = _load_config()
    mcfg = cfg["active"]["model"]
    model, tok = load_model_for_inference(
        mcfg["name"],
        adapter_path=str(adapter_path),
        use_4bit=mcfg.get("use_4bit", True),
    )
    holdout = ROOT / "data" / f"holdout_{cfg['active']['data']['holdout_size']}.jsonl"
    with open(holdout, encoding="utf-8") as f:
        item = _json.loads(f.readline())
    preview = generate_answer(
        model, tok, item["instruction"], max_new_tokens=max_new_tokens
    )
    del model, tok
    if "!" * 20 in preview or preview.strip() in ("!", "!!", "!!!"):
        raise RuntimeError(f"生成崩溃（重复!）: {preview[:80]!r}")
    return preview


def backup_broken_e3() -> None:
    cfg = load_config()
    out = output_dir(cfg)
    src = out / "E3_size10000_lr2e-4"
    if not src.exists():
        log("E3 旧目录不存在，跳过备份")
        return
    dst = out / "E3_size10000_lr2e-4_broken_2epoch"
    if dst.exists():
        log(f"备份已存在: {dst}")
        return
    src.rename(dst)
    log(f"已备份旧 E3 → {dst}")


def train_and_eval_e3(lr: float) -> Path:
    exp_id, size = "E3", 10000
    py = sys.executable

    run_step(
        f"E3 训练 lr={lr}",
        [py, "scripts/train_lora.py", "--train_size", str(size), "--exp_id", exp_id, "--learning_rate", str(lr)],
    )

    adapter = adapter_dir(exp_id, size, lr)
    if not adapter.exists():
        raise FileNotFoundError(f"训练完成但 adapter 不存在: {adapter}")

    log("E3 生成抽查 …")
    preview = sanity_check_adapter(adapter)
    log(f"E3 抽查 OK，预览: {preview[:120]}")

    run_step(
        "E3 C-Eval",
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
        "E3 Hold-out",
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
    return adapter


def write_ablation_summary() -> Path:
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
    log("=== Ablation 汇总 ===")
    log(summary.to_string(index=False))
    log(f"已保存: {out_path}")
    return out_path


def write_report(status: dict) -> None:
    lines = [
        f"夜间流水线报告  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        json.dumps(status, ensure_ascii=False, indent=2),
        "",
        "日志: logs/overnight_latest.log",
        "汇总: results/ablation_summary.csv",
    ]
    text = "\n".join(lines)
    REPORT_FILE.write_text(text, encoding="utf-8")
    log(f"报告已写入: {REPORT_FILE}")


def main() -> None:
    if LOG_FILE.exists():
        LOG_FILE.unlink()
    log("夜间流水线启动")
    log(f"工作目录: {ROOT}")

    py = sys.executable
    status: dict = {"steps": {}, "success": False, "e3_adapter": None}

    try:
        run_step(
            "预检",
            [py, "scripts/preflight_check.py"],
        )
        status["steps"]["preflight"] = "ok"

        e2_adapter = adapter_dir("E2", 5000, 2e-4)
        if not e2_adapter.exists():
            raise FileNotFoundError(f"E2 adapter 不存在: {e2_adapter}")

        run_step(
            "E2 C-Eval",
            [
                py,
                "scripts/evaluate_ceval.py",
                "--mode",
                "lora",
                "--adapter",
                str(e2_adapter),
                "--exp_id",
                "E2",
            ],
            infer_gpu=True,
        )
        run_step(
            "E2 Hold-out",
            [
                py,
                "scripts/evaluate_holdout.py",
                "--adapter",
                str(e2_adapter),
                "--exp_id",
                "E2",
            ],
            infer_gpu=True,
        )
        bang, score = holdout_bang_ratio("E2")
        status["steps"]["E2_eval_old"] = {"bang_ratio": bang, "lora_score_mean": score}
        log(f"E2 旧 adapter 检查: bang_ratio={bang:.2%}, lora_score_mean={score:.4f}")
        if bang > 0.1:
            log("E2 旧 adapter 已崩溃，将重训 E2（1 epoch）…")
            import shutil

            broken = output_dir(load_config()) / "E2_size5000_lr2e-4_broken_2epoch"
            src = output_dir(load_config()) / "E2_size5000_lr2e-4"
            if src.exists() and not broken.exists():
                shutil.move(str(src), str(broken))
            run_step(
                "E2 重训",
                [py, "scripts/train_lora.py", "--train_size", "5000", "--exp_id", "E2"],
            )
            e2_adapter = adapter_dir("E2", 5000, 2e-4)
            sanity_check_adapter(e2_adapter)
            run_step(
                "E2 C-Eval (重训后)",
                [py, "scripts/evaluate_ceval.py", "--mode", "lora", "--adapter", str(e2_adapter), "--exp_id", "E2"],
                infer_gpu=True,
            )
            run_step(
                "E2 Hold-out (重训后)",
                [py, "scripts/evaluate_holdout.py", "--adapter", str(e2_adapter), "--exp_id", "E2"],
                infer_gpu=True,
            )
            bang, score = holdout_bang_ratio("E2")
            status["steps"]["E2_retrain"] = {"bang_ratio": bang, "lora_score_mean": score}
            if bang > 0.1:
                raise RuntimeError(f"E2 重训后仍异常（{bang:.0%} 为纯!），请尝试 --learning_rate 1e-4")

        backup_broken_e3()
        status["steps"]["E3_backup"] = "ok"

        try:
            adapter = train_and_eval_e3(lr=2e-4)
            status["steps"]["E3"] = {"lr": 2e-4, "status": "ok"}
        except Exception as exc:
            log(f"E3 lr=2e-4 失败: {exc}")
            log("回退重试 E3 lr=1e-4 …")
            status["steps"]["E3_first_try"] = {"lr": 2e-4, "error": str(exc)}
            adapter = train_and_eval_e3(lr=1e-4)
            status["steps"]["E3"] = {"lr": 1e-4, "status": "ok_retry"}

        status["e3_adapter"] = str(adapter)
        bang, score = holdout_bang_ratio("E3")
        status["steps"]["E3_holdout_check"] = {"bang_ratio": bang, "lora_score_mean": score}

        write_ablation_summary()
        status["success"] = True
        log("夜间流水线全部完成 ✓")

    except Exception as exc:
        log(f"流水线失败: {exc}")
        log(traceback.format_exc())
        status["error"] = str(exc)
        status["success"] = False
    finally:
        write_report(status)
        if not status["success"]:
            sys.exit(1)


if __name__ == "__main__":
    main()
