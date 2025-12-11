#!/usr/bin/env python3
# coding: utf-8入 {file_name, text} 记录。
import os
import json
from pathlib import Path

# -------- 配置区（如需修改只改这里） --------
ROOT_DIR = Path(
    "/lab310/enhance/BioDiffuse_Base/BBBC/Human Hepatocyte and Murine Fibroblast cells – Co-culture experiment"
)
OUT_PATH = Path("/data0/syhong/BioDiff/dataset/1_foundation_model/BBBC_hepa_fibro_dna.jsonl")
PROMPT = "DNA of Human Hepatocyte and Murine Fibroblast cells"
# --------------------------------------------

def is_tif(file_path: Path) -> bool:
    return file_path.suffix.lower() in {".png"}

def main() -> None:
    n_total, n_written = 0, 0
    with OUT_PATH.open("w", encoding="utf-8") as fout:
        for file_path in ROOT_DIR.rglob("*"):
            if not file_path.is_file():
                continue
            if not is_tif(file_path):
                continue

            n_total += 1
            record = {
                "file_name": str(file_path.resolve()),
                "text": PROMPT
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"{n_total} .tif file, {n_written} → {OUT_PATH}")

if __name__ == "__main__":
    main()
