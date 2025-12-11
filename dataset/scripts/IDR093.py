#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path

ROOT_DIR = Path(
    "/lab310/enhance/BioDiffuse_Base/IDR_raw/idr0093-mueller-perturbation"
)
OUT_FILE = Path(
    "/data0/syhong/BioDiff/dataset/1_foundation_model/idr093.jsonl"
)
PROMPT_RULES = {
    "c01": "DNA of HeLa Kyoto",
    "c02": "Nascent RNA of HeLa Kyoto",
    "c03": "PCNA of HeLa Kyoto",
    "c04": "Succinimidyl ester of HeLa Kyoto",
}


def is_tif(path: Path) -> bool:
    return path.suffix.lower() in {".tif", ".tiff"}


def infer_prompt(filename: str) -> str | None:
    stem = Path(filename).stem.lower()
    for suffix, prompt in PROMPT_RULES.items():
        if stem.endswith(suffix):
            return prompt
    return None


def main() -> None:
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    total, written = 0, 0
    with OUT_FILE.open("w", encoding="utf-8") as fout:
        for path in ROOT_DIR.rglob("*"):
            if not path.is_file() or not is_tif(path):
                continue
            total += 1

            prompt = infer_prompt(path.name)
            if prompt is None:
                continue

            record = {
                "file_name": str(path.resolve()),
                "text": prompt,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(
        f"{total} tif，"
        f"{written}  → {OUT_FILE}"
    )


if __name__ == "__main__":
    main()
