#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create JSONL prompt file for IDR088 tif images.

Each line:
{"file_name": "<abs_path>", "text": "<organelle> of <cell_line>"}
"""

from __future__ import annotations

import json
import re
from pathlib import Path
import pandas as pd

CSV_PATH = Path(
    "/data0/syhong/BioDiff/dataset/4_annotation_files/idr0088-screenA-annotation.csv"
)
IMG_ROOT = Path(
    "/lab310/enhance/BioDiffuse_Base/IDR_raw/IDR088"
)
OUT_FILE = Path(
    "/data0/syhong/BioDiff/dataset/1_foundation_model/idr088.jsonl"
)

CH_RE = re.compile(r"C(\d{2})", re.IGNORECASE)
# 解析 “Ch2 (green): ACTB” 这种字段
CHANNEL_DEF_RE = re.compile(r"Ch(\d+).*?:\s*([^,]+)", re.IGNORECASE)

def load_annotations(csv_path: Path):
    df = pd.read_csv(csv_path, sep=None, engine="python")
    index: dict[tuple[str, str], dict] = {}

    for _, row in df.iterrows():
        plate_loc_full = str(row["Plate Location"]).strip()
        plate_loc = plate_loc_full.split("/")[-1]

        plate = str(row["Plate"]).strip()
        cell_line = str(row["Characteristics [Cell Line]"]).strip()

        chan_map: dict[int, str] = {}
        for m in CHANNEL_DEF_RE.finditer(str(row["Channels"])):
            chan_map[int(m.group(1))] = m.group(2).strip()

        index[(plate_loc, plate)] = {"cell_line": cell_line, "chan_map": chan_map}
    return index

def main() -> None:
    anno = load_annotations(CSV_PATH)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    scanned = written = 0
    with OUT_FILE.open("w", encoding="utf-8") as fout:
        for img_path in IMG_ROOT.rglob("*"):
            if img_path.suffix.lower() not in {".tif", ".tiff"}:
                continue
            scanned += 1

            try:
                rel_parts = img_path.relative_to(IMG_ROOT).parts
                plate_location, plate = rel_parts[0], rel_parts[1]
            except (ValueError, IndexError):
                continue

            key = (plate_location, plate)
            if key not in anno:
                continue

            m = CH_RE.search(img_path.stem)
            if not m:
                continue
            ch_num = int(m.group(1).lstrip("0") or "0")  # '02' -> 2

            chan_map = anno[key]["chan_map"]
            organelle = chan_map.get(ch_num)
            if organelle is None:
                continue

            prompt = f"{organelle} of {anno[key]['cell_line']}"
            record = {"file_name": str(img_path.resolve()), "text": prompt}
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"scan {scanned} tif, {written} → {OUT_FILE}")


if __name__ == "__main__":
    main()
