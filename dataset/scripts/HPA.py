#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate JSONL prompts for HPAv18 JPG images.

Output example:
{"file_name": "/abs/path/.../10580_1610_C1_1_green.jpg", "text": "Nucleoplasm and Cytosol of RH-30"}
"""

import csv
import json
import re
from pathlib import Path
import pandas as pd

CSV_PATH = Path(
    "/data0/syhong/BioDiff/dataset/4_annotation_files/HPAv18RBG_wodpl_withCellLine.csv"
)
IMG_ROOT = Path(
    "/lab310/enhance/BioDiffuse_Base/HPAv18"
)
OUT_FILE = Path(
    "/data0/syhong/BioDiff/dataset/1_foundation_model/HPA.jsonl"
)
CODE2ORG = [
    "Nucleoplasm", "Nuclear membrane", "Nucleoli", "Nucleoli fibrillar center",
    "Nuclear speckles", "Nuclear bodies", "Endoplasmic reticulum",
    "Golgi apparatus", "Peroxisomes", "Endosomes", "Lysosomes",
    "Intermediate filaments", "Actin filaments", "Focal adhesion sites",
    "Microtubules", "Microtubule ends", "Cytokinetic bridge",
    "Mitotic spindle", "Microtubule organizing center", "Centrosome",
    "Lipid droplets", "Plasma membrane", "Cell junctions", "Mitochondria",
    "Aggresome", "Cytosol", "Cytoplasmic bodies", "Rods & rings"
]
CHANNEL_PROMPT = {
    "blue":  "nucleus",
    "red":   "microtubules",
    "yellow": "er",
}
CH_PATTERN = re.compile(r"_(blue|green|red|yellow)$", re.IGNORECASE)
# ------------------------------------


def load_annotation(csv_path: str) -> dict[str, tuple[list[int], str]]:
    """
    读取注释表，返回  {Id: ([target_code...], cell_line)}
    """
    df = pd.read_csv(csv_path, sep=None, engine="python")
    mapping = {}
    for _, row in df.iterrows():
        id_ = str(row["Id"]).strip()
        codes = [int(c) for c in str(row["Target"]).split()]
        cell_line = str(row["CellLine"]).strip()
        mapping[id_] = (codes, cell_line)
    return mapping


def codes_to_orgs(codes: list[int]) -> str:
    names = [CODE2ORG[c] for c in codes]
    return " and ".join(names)


def main() -> None:
    id2info = load_annotation(CSV_PATH)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    scanned = written = 0
    with OUT_FILE.open("w", encoding="utf-8") as fout:
        for img_path in IMG_ROOT.rglob("*.jpg"):
            scanned += 1
            stem = img_path.stem
            m = CH_PATTERN.search(stem)
            if not m:
                continue
            channel = m.group(1).lower()  # blue / green / red / yellow
            img_id = CH_PATTERN.sub("", stem) 

            if img_id not in id2info:
                continue
            codes, cell_line = id2info[img_id]

            if channel == "green":
                organelle = codes_to_orgs(codes)
            else:
                organelle = CHANNEL_PROMPT[channel]

            prompt = f"{organelle} of {cell_line}"
            record = {
                "file_name": str(img_path.resolve()),
                "text": prompt,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"{scanned} jpg，writen {written}  → {OUT_FILE}")


if __name__ == "__main__":
    main()
