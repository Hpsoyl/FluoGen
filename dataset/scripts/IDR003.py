#!/usr/bin/env python3
# coding: utf-8

import os
import json
from pathlib import Path

ROOT_DIR = Path(
    "/lab310/enhance/BioDiffuse_Base/IDR_raw/idr0003-breker-plasticity"
)
OUT_DIR = Path("/data0/syhong/BioDiff/dataset/1_foundation_model")
OUT_FILE = OUT_DIR / "idr003.jsonl"

PROMPT_RULES = {
    "cherry": "H2B-mCherry of Y6545",
    "gfp": "GFP of Y6545",
}
# ----------------------------------------

def is_tif(file_path: Path) -> bool:
    return file_path.suffix.lower() in {".tif", ".tiff"}

def infer_prompt(filename: str) -> str | None:
    stem_lower = Path(filename).stem.lower()
    for suffix, prompt in PROMPT_RULES.items():
        if stem_lower.endswith(suffix):
            return prompt
    return None

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    n_scanned = n_written = 0
    with OUT_FILE.open("w", encoding="utf-8") as fout:
        for path in ROOT_DIR.rglob("*"):
            if not path.is_file() or not is_tif(path):
                continue
            n_scanned += 1

            prompt = infer_prompt(path.name)
            if prompt is None:
                continue

            record = {
                "file_name": str(path.resolve()),
                "text": prompt
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"{n_scanned} tif，{n_written} satisfy the rules ")
    print(f"been writen → {OUT_FILE}")

if __name__ == "__main__":
    main()
