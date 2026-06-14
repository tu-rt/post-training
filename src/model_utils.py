from __future__ import annotations

import gc
import re
from typing import Optional

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.config import resolve_model_path, validate_local_model


def _model_input_device(model) -> torch.device:
    """多卡 device_map='auto' 时，输入应放到 embedding 所在设备。"""
    if hasattr(model, "get_input_embeddings"):
        return model.get_input_embeddings().weight.device
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def load_tokenizer(model_name: str):
    model_name = resolve_model_path(model_name)
    validate_local_model(model_name)
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base_model(
    model_name: str,
    use_4bit: bool = True,
    gradient_checkpointing: bool = True,
):
    kwargs: dict = {"trust_remote_code": True}
    if use_4bit and torch.cuda.is_available():
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        kwargs["device_map"] = "auto"
    else:
        kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
        kwargs["device_map"] = "auto" if torch.cuda.is_available() else None

    model_name = resolve_model_path(model_name)
    validate_local_model(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if use_4bit and torch.cuda.is_available():
        model = prepare_model_for_kbit_training(model)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    return model


def build_lora_model(
    model_name: str,
    lora_cfg: dict,
    use_4bit: bool = True,
    gradient_checkpointing: bool = True,
):
    model = load_base_model(model_name, use_4bit, gradient_checkpointing)
    peft_cfg = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, peft_cfg)


def load_model_for_inference(
    model_name: str,
    adapter_path: Optional[str] = None,
    use_4bit: bool = True,
):
    tokenizer = load_tokenizer(model_name)
    model = load_base_model(model_name, use_4bit=use_4bit, gradient_checkpointing=False)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def generate_answer(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    max_input_tokens: int | None = None,
) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    if max_input_tokens is None:
        max_input_tokens = getattr(tokenizer, "model_max_length", 2048)
    # 为 max_new_tokens 预留空间，避免超长 prompt 导致 logits 数值异常
    max_input_tokens = max(64, max_input_tokens - max_new_tokens)
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    )
    device = _model_input_device(model)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    gen_kwargs: dict = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        gen_kwargs.update(
            do_sample=True,
            temperature=max(temperature, 1e-5),
            top_p=0.9,
        )
    else:
        gen_kwargs["do_sample"] = False

    out = model.generate(**inputs, **gen_kwargs)
    gen_ids = out[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def extract_choice_letter(text: str) -> str | None:
    text = text.strip().upper()
    m = re.search(r"([ABCD])", text)
    return m.group(1) if m else None


def free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
