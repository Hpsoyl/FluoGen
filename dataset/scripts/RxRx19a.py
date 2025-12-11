#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import re
from pathlib import Path

ROOT_DIR = Path(
    "/lab310/enhance/BioDiffuse_Base/RxRx/RxRx19a/images"
)
OUT_FILE = Path(
    "/data0/syhong/BioDiff/dataset/1_foundation_model/rxrx19a.jsonl"
)

CHANNEL2ORG = {
    "w1": "nuclei",
    "w2": "er",
    "w3": "actin",
    "w4": "nucleoli and cytoplasmic RNA",
    "w5": "golgi",
}
W_PATTERN = re.compile(r"_w([1-5])$", re.IGNORECASE)
# --------------------------------------------


def is_png(path: Path) -> bool:
    return path.suffix.lower() == ".png"


def get_cell_name(batch_dir: str) -> str:
    return batch_dir.split("-")[0].lower()


def infer_organelle(stem: str) -> str | None:
    m = W_PATTERN.search(stem)
    if not m:
        return None
    key = f"w{m.group(1)}".lower()
    return CHANNEL2ORG.get(key)


def main() -> None:
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    total, written = 0, 0
    with OUT_FILE.open("w", encoding="utf-8") as fout:
        for img_path in ROOT_DIR.rglob("*.png"):
            if not is_png(img_path):
                continue
            total += 1

            try:
                batch_dir = img_path.relative_to(ROOT_DIR).parts[0]
            except ValueError:
                continue

            cell_name = get_cell_name(batch_dir)
            organelle = infer_organelle(img_path.stem)
            if organelle is None:
                continue

            prompt = f"{organelle} of {cell_name}"
            record = {
                "file_name": str(img_path.resolve()),
                "text": prompt,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(
        f"{total}  PNG, {written} → {OUT_FILE}"
    )


if __name__ == "__main__":
    main()
