from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from datasets import load_dataset


def _clean_text(text: str, min_chars: int, max_chars: int) -> str | None:
    text = (text or "").strip()
    text = re.sub(r"\s+", " ", text)
    if len(text) < min_chars:
        return None
    if len(text) > max_chars:
        text = text[:max_chars]
    return text


def _extract_instruction_response(item: dict[str, Any]) -> tuple[str, str] | None:
    """兼容 COIG-CQIA 多种字段命名。"""
    instruction = (
        item.get("instruction")
        or item.get("input")
        or item.get("question")
        or item.get("prompt")
        or ""
    )
    response = (
        item.get("output")
        or item.get("response")
        or item.get("answer")
        or item.get("content")
        or ""
    )
    if not instruction and not response:
        conv = item.get("conversation") or item.get("conversations")
        if isinstance(conv, list) and len(conv) >= 2:
            instruction = conv[0].get("value") or conv[0].get("content") or ""
            response = conv[1].get("value") or conv[1].get("content") or ""
    instruction = str(instruction).strip()
    extra = item.get("input") or item.get("query") or ""
    extra = str(extra).strip()
    if extra and extra not in instruction:
        instruction = f"{instruction}\n{extra}" if instruction else extra
    response = str(response).strip()
    if not instruction or not response:
        return None
    return instruction, response


def to_messages(instruction: str, response: str) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": response},
    ]


def load_coig_cqia(
    source: str,
    seed: int,
    min_chars: int,
    max_chars: int,
    subset: str | list[str] | None = None,
    cache_path: Path | None = None,
) -> list[dict[str, Any]]:
    if cache_path and cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)

    if isinstance(subset, list):
        ds_list = [load_dataset(source, name=sub, split="train") for sub in subset]
        from datasets import concatenate_datasets
        ds = concatenate_datasets(ds_list)
    elif subset:
        ds = load_dataset(source, name=subset, split="train")
    else:
        ds = load_dataset(source, split="train")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    for item in ds:
        pair = _extract_instruction_response(item)
        if not pair:
            continue
        inst, resp = pair
        inst = _clean_text(inst, min_chars, max_chars)
        resp = _clean_text(resp, min_chars, max_chars)
        if not inst or not resp:
            continue
        key = inst + "||" + resp
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "instruction": inst,
                "response": resp,
                "messages": to_messages(inst, resp),
            }
        )

    rng = __import__("random").Random(seed)
    rng.shuffle(rows)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False)

    return rows


def split_train_holdout(
    rows: list[dict[str, Any]],
    train_size: int,
    holdout_size: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    total_needed = train_size + holdout_size
    if len(rows) < total_needed:
        raise ValueError(
            f"数据不足：需要 {total_needed} 条，当前仅 {len(rows)} 条。"
            "请减小 train_size / holdout_size 或检查数据源。"
        )
    pool = rows[:total_needed]
    holdout = pool[:holdout_size]
    train = pool[holdout_size : holdout_size + train_size]
    return train, holdout


def save_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
