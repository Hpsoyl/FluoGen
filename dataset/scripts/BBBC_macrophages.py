#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json

ROOT_DIR = "/lab310/enhance/BioDiffuse_Base/BBBC/Murine bone-marrow derived macrophages"
OUT_PATH = "/data0/syhong/BioDiff/dataset/1_foundation_model/BBBC_macrophages.jsonl"

PROMPT_RULES = {
    "c1": "CD11b of Bone Marrow-Derived Macrophages from C57BL/6 mice",
    "c5": "nuclei of Bone Marrow-Derived Macrophages from C57BL/6 mice",
}

def infer_prompt(filename: str) -> str | None:
    base = os.path.splitext(filename)[0].lower()
    for suffix, prompt in PROMPT_RULES.items():
        if base.endswith(suffix):
            return prompt
    return None

def main() -> None:
    n_written = 0
    with open(OUT_PATH, "w", encoding="utf-8") as f_out:
        for dirpath, _, filenames in os.walk(ROOT_DIR):
            for fname in filenames:
                if not fname.lower().endswith(".tif"):
                    continue

                prompt = infer_prompt(fname)
                if prompt is None:
                    continue

                abs_path = os.path.join(dirpath, fname)
                record = {"file_name": abs_path, "text": prompt}
                json_line = json.dumps(record, ensure_ascii=False)
                f_out.write(json_line + "\n")
                n_written += 1

    print(f"total {n_written} → {OUT_PATH}")

if __name__ == "__main__":
    main()
