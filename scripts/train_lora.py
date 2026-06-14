"""
QLoRA 监督微调（SFT）。

用法:
  python scripts/train_lora.py --train_size 200 --exp_id E1
  python scripts/train_lora.py --train_size 200 --exp_id E1 --max_steps 30   # 冒烟
"""
from __future__ import annotations

import os
import sys

# 必须在 import torch 之前设置。7B QLoRA 单张 4090 足够；
# 多卡 device_map="auto" 分片训练会触发 cuda:0/1/3 跨设备 loss 报错（问题 #12）
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("TRAIN_CUDA_VISIBLE_DEVICES", "0"))
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")

import argparse
import inspect
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.adapter_utils import check_adapter_weights
from src.config import data_dir, get_num_train_epochs, load_config, output_dir, print_run_context
from src.model_utils import load_base_model, load_tokenizer


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_sft_config(seq_length: int, **kwargs) -> SFTConfig:
    """兼容 trl 0.12.x (max_seq_length) 与较新版 (max_length)。"""
    sig = inspect.signature(SFTConfig.__init__)
    if "max_seq_length" in sig.parameters:
        kwargs["max_seq_length"] = seq_length
    elif "max_length" in sig.parameters:
        kwargs["max_length"] = seq_length
    else:
        print("[WARN] 当前 trl 的 SFTConfig 不支持序列长度参数，将使用默认截断")

    allowed = {k: v for k, v in kwargs.items() if k in sig.parameters}
    dropped = set(kwargs) - set(allowed) - {"max_seq_length", "max_length"}
    if dropped:
        print(f"[WARN] SFTConfig 忽略不兼容参数: {sorted(dropped)}")
    return SFTConfig(**allowed)


def build_sft_trainer(model, training_args, train_dataset, tokenizer, formatting_func):
    """兼容 trl 0.9.x ~ 0.12.x 的 SFTTrainer 参数名差异（问题 #11）。"""
    sig = inspect.signature(SFTTrainer.__init__)
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "formatting_func": formatting_func,
    }
    if "processing_class" in sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    return SFTTrainer(**trainer_kwargs)


def warn_if_loss_abnormal(log_history: list[dict]) -> None:
    losses = [entry["loss"] for entry in log_history if "loss" in entry]
    if not losses:
        return
    final_loss = losses[-1]
    min_loss = min(losses)
    print(f"[train] 最终 loss={final_loss:.4f}, 最低 loss={min_loss:.4f}")
    if final_loss < 0.3 or min_loss < 0.1:
        print(
            "[WARN] loss 异常偏低，可能存在过拟合，评测时生成可能出现 nan。"
            "建议减少 epoch 或降低 learning_rate 后重训。"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--train_size", type=int, required=True)
    parser.add_argument("--exp_id", default="E1")
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--train_file", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cuda_dev = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
    print_run_context(
        cfg,
        script="train_lora",
        extra=f"exp_id={args.exp_id}, train_size={args.train_size}, cuda={cuda_dev}",
    )
    active = cfg["active"]
    mcfg = active["model"]
    lcfg = active["lora"]
    tcfg = active["train"]

    train_path = (
        Path(args.train_file)
        if args.train_file
        else data_dir(cfg) / f"train_{args.train_size}.jsonl"
    )
    if not train_path.exists():
        raise FileNotFoundError(
            f"找不到 {train_path}，请先运行: "
            f"python scripts/prepare_data.py --train_size {args.train_size}"
            " 或 python scripts/prepare_data.py --all_sizes"
        )

    rows = load_jsonl(train_path)
    print(f"训练样本: {len(rows)} 来自 {train_path}")

    model_name = mcfg["name"]
    use_4bit = mcfg.get("use_4bit", True)
    tokenizer = load_tokenizer(model_name)

    try:
        model = load_base_model(
            model_name,
            use_4bit=use_4bit,
            gradient_checkpointing=tcfg.get("gradient_checkpointing", True),
        )
    except Exception as exc:
        fallback = mcfg.get("fallback")
        if not fallback:
            raise
        print(f"[train] 主模型加载失败: {exc}，回退 {fallback}")
        model_name = fallback
        use_4bit = mcfg.get("use_4bit_fallback", False)
        tokenizer = load_tokenizer(model_name)
        model = load_base_model(
            model_name,
            use_4bit=use_4bit,
            gradient_checkpointing=tcfg.get("gradient_checkpointing", True),
        )

    peft_cfg = LoraConfig(
        r=lcfg["r"],
        lora_alpha=lcfg["alpha"],
        lora_dropout=lcfg["dropout"],
        target_modules=lcfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    ds = Dataset.from_list(rows)

    lr = args.learning_rate if args.learning_rate is not None else tcfg["learning_rate"]
    max_steps = args.max_steps if args.max_steps is not None else tcfg.get("max_steps", -1)
    num_epochs = get_num_train_epochs(tcfg, args.train_size)
    print(f"[train] num_train_epochs={num_epochs} (train_size={args.train_size})")

    lr_tag = f"{lr:.0e}".replace("e-0", "e-").replace("e+", "e")
    run_name = f"{args.exp_id}_size{args.train_size}_lr{lr_tag}"
    save_dir = output_dir(cfg) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    training_args = build_sft_config(
        mcfg["max_seq_length"],
        output_dir=str(save_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=tcfg["per_device_train_batch_size"],
        gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
        learning_rate=lr,
        warmup_ratio=tcfg["warmup_ratio"],
        logging_steps=tcfg["logging_steps"],
        save_steps=tcfg["save_steps"],
        save_total_limit=2,
        max_steps=max_steps,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        optim="paged_adamw_8bit" if use_4bit else "adamw_torch",
        max_grad_norm=tcfg.get("max_grad_norm", 1.0),
        lr_scheduler_type="cosine",
        report_to="none",
        dataset_text_field="text",
        packing=False,
        gradient_checkpointing=tcfg.get("gradient_checkpointing", True),
    )

    def formatting_func(example: dict) -> str:
        return tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )

    trainer = build_sft_trainer(model, training_args, ds, tokenizer, formatting_func)

    print(f"开始训练 → {save_dir}")
    trainer.train()
    warn_if_loss_abnormal(trainer.state.log_history)

    adapter_path = save_dir / "adapter"
    trainer.save_model(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    check_adapter_weights(adapter_path)
    print(f"[OK] adapter 权重检查通过: {adapter_path}")

    meta = {
        "exp_id": args.exp_id,
        "train_size": args.train_size,
        "learning_rate": lr,
        "num_train_epochs": num_epochs,
        "base_model": model_name,
        "adapter_path": str(adapter_path.resolve()),
        "max_steps": max_steps,
        "profile": cfg["active_profile_name"],
        "final_loss": trainer.state.log_history[-1].get("loss") if trainer.state.log_history else None,
    }
    with open(save_dir / "train_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"训练完成，adapter 保存在: {adapter_path}")


if __name__ == "__main__":
    main()
