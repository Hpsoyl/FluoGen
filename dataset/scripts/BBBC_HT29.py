#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为 HT29 shRNAi screen 数据集生成 prompt-jsonl
"""

import json
from pathlib import Path

ROOT_DIR = Path(
    "/lab310/enhance/BioDiffuse_Base/BBBC/Human_HT29_colon-cancer_cells_shRNAi_screen_converted"
)
OUT_FILE = Path(
    "/data0/syhong/BioDiff/dataset/1_foundation_model/BBBC_ht29.jsonl"
)
SUFFIX2PROMPT = {
    "channel1": "DNA of ht29",
    "channel2": "pH3 of ht29",
    "channel3": "actin of ht29",
}

def is_tif(path: Path) -> bool:
    return path.suffix.lower() in {".tif", ".tiff"}

def infer_prompt(stem: str) -> str | None:
    for suffix, prompt in SUFFIX2PROMPT.items():
        if stem.lower().endswith(suffix):
            return prompt
    return None

def main() -> None:
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    scanned, written = 0, 0
    with OUT_FILE.open("w", encoding="utf-8") as fout:
        for img_path in ROOT_DIR.rglob("*"):
            if not img_path.is_file() or not is_tif(img_path):
                continue
            scanned += 1

            prompt = infer_prompt(img_path.stem)
            if prompt is None:
                continue

            record = {
                "file_name": str(img_path.resolve()),
                "text": prompt,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"{scanned} tif, {written}  → {OUT_FILE}")

if __name__ == "__main__":
    main()
