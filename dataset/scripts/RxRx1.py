#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
from pathlib import Path

ROOT_DIR = Path(
    "/lab310/enhance/BioDiffuse_Base/RxRx/rxrx1/images"
)
OUT_FILE = Path(
    "/data0/syhong/BioDiff/dataset/1_foundation_model/rxrx1.jsonl"
)

CHANNEL2ORGANELLE = {
    "w1": "nuclei",
    "w2": "er",
    "w3": "actin",
    "w4": "nucleoli and cytoplasmic RNA",
    "w5": "mitochondria",
    "w6": "golgi",
}
# ------------------------------

PNG_PATTERN = re.compile(r"_w([1-6])$", re.IGNORECASE)


def is_png(path: Path) -> bool:
    return path.suffix.lower() == ".png"


def get_cell_name(cell_batch_dir: str) -> str:
    return cell_batch_dir.split("-")[0].lower()


def infer_organell_from_stem(stem: str) -> str | None:
    m = PNG_PATTERN.search(stem)
    if not m:
        return None
    key = f"w{m.group(1)}"
    return CHANNEL2ORGANELLE.get(key.lower())


def main() -> None:
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    n_total, n_written = 0, 0
    with OUT_FILE.open("w", encoding="utf-8") as fout:
        for img_path in ROOT_DIR.rglob("*.png"):
            if not is_png(img_path):
                continue
            n_total += 1

            try:
                cell_batch_dir = img_path.relative_to(ROOT_DIR).parts[0]
            except ValueError:  # 理论上不会出现
                continue
            cell_name = get_cell_name(cell_batch_dir)

            organelle = infer_organell_from_stem(img_path.stem)
            if organelle is None:
                continue

            prompt = f"{organelle} of {cell_name}"
            record = {
                "file_name": str(img_path.resolve()),
                "text": prompt,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"{n_total} PNG, {n_written} → {OUT_FILE}")


if __name__ == "__main__":
    main()
