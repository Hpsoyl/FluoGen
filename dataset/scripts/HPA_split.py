import os
import json

input_path = "/data0/syhong/BioDiff/dataset/1_foundation_model/HPA.jsonl"
output_dir = "/data0/syhong/BioDiff/dataset/5_HPA_split"
os.makedirs(output_dir, exist_ok=True)

organelles = {}
with open(input_path, "r", encoding="utf-8") as infile:
    for line in infile:
        try:
            entry = json.loads(line.strip())
            text = entry.get("text", "")
            if "and" in text.lower():
                continue
            if " of " not in text:
                continue

            organelle = text.split(" of ")[0].strip().lower().replace(" ", "_")
            organelles.setdefault(organelle, []).append(entry)

        except json.JSONDecodeError:
            continue

for organelle, entries in organelles.items():
    output_path = os.path.join(output_dir, f"HPA_{organelle}.jsonl")
    with open(output_path, "w", encoding="utf-8") as outfile:
        for item in entries:
            json.dump(item, outfile)
            outfile.write("\n")

print(f"{len(organelles)} organelle categories has been splited, and been writen into {output_dir}")
