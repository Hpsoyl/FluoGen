#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path

ROOT_DIR = Path(
    "/lab310/enhance/BioDiffuse_Base/IDR_raw/idr0094-ellinger-sarscov2"
)
OUT_FILE = Path(
    "/data0/syhong/BioDiff/dataset/1_foundation_model/idr094.jsonl"
)
PROMPT_TEXT = "caco2"
# ---------------------------


def is_tif(path: Path) -> bool:
    return path.suffix.lower() in {".tif", ".tiff"}


def main() -> None:
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    total, written = 0, 0
    with OUT_FILE.open("w", encoding="utf-8") as fout:
        for path in ROOT_DIR.rglob("*"):
            if not path.is_file() or not is_tif(path):
                continue
            total += 1

            record = {
                "file_name": str(path.resolve()),
                "text": PROMPT_TEXT,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(
        f"{total} tif, {written} → {OUT_FILE}"
    )


if __name__ == "__main__":
    main()
