from __future__ import annotations

from pathlib import Path

import torch


def check_adapter_weights(adapter_path: str | Path) -> None:
    """训练/评测前检查 LoRA 权重是否含 nan/inf。"""
    path = Path(adapter_path)
    if not path.exists():
        raise FileNotFoundError(f"adapter 不存在: {path}")

    weight_files = list(path.glob("*.safetensors")) + list(path.glob("adapter_model.bin"))
    if not weight_files:
        raise FileNotFoundError(f"adapter 目录无权重文件: {path}")

    for wf in weight_files:
        if wf.suffix == ".safetensors":
            from safetensors.torch import load_file

            weights = load_file(str(wf))
        else:
            try:
                weights = torch.load(wf, map_location="cpu", weights_only=True)
            except TypeError:
                weights = torch.load(wf, map_location="cpu")

        for name, tensor in weights.items():
            if not torch.is_floating_point(tensor):
                continue
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                raise RuntimeError(
                    f"adapter 权重含 nan/inf: {name} ({wf.name})。"
                    "建议降低 learning_rate / 减少 epoch 后重新训练。"
                )
