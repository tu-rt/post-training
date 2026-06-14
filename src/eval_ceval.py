from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset


CEVAL_SUBJECT_ZH = {
    "computer_network": "计算机网络",
    "operating_system": "操作系统",
    "discrete_mathematics": "离散数学",
    "probability_and_statistics": "概率统计",
    "college_chemistry": "大学化学",
}


def build_ceval_prompt(subject: str, item: dict[str, Any]) -> tuple[str, str]:
    subj_zh = CEVAL_SUBJECT_ZH.get(subject, subject)
    q = item["question"]
    prompt = (
        f"以下是中国关于{subj_zh}考试的单项选择题，请选出正确答案。\n\n"
        f"题目：{q}\n"
        f"A. {item['A']}\n"
        f"B. {item['B']}\n"
        f"C. {item['C']}\n"
        f"D. {item['D']}\n\n"
        "请只回答选项字母（A/B/C/D）："
    )
    gold = str(item["answer"]).strip().upper()
    return prompt, gold


def load_ceval_subset(
    dataset_name: str,
    subjects: list[str],
    per_subject: int,
    seed: int = 42,
    cache_path: Path | None = None,
) -> list[dict[str, Any]]:
    if cache_path and cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)

    rows: list[dict[str, Any]] = []
    for subject in subjects:
        try:
            ds = load_dataset(dataset_name, subject, split="val")
        except Exception:
            ds = load_dataset(dataset_name, subject, split="test")
        ds = ds.shuffle(seed=seed).select(range(min(per_subject, len(ds))))
        for i, item in enumerate(ds):
            prompt, gold = build_ceval_prompt(subject, item)
            rows.append(
                {
                    "id": f"{subject}_{i}",
                    "subject": subject,
                    "prompt": prompt,
                    "gold": gold,
                }
            )

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
    return rows


def summarize_ceval(records: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    if df.empty:
        return df
    overall = {
        "subject": "overall",
        "accuracy": round(df["correct"].mean(), 4),
        "n": len(df),
    }
    by_subj = (
        df.groupby("subject", as_index=False)
        .agg(accuracy=("correct", "mean"), n=("correct", "count"))
        .round(4)
    )
    return pd.concat([by_subj, pd.DataFrame([overall])], ignore_index=True)
