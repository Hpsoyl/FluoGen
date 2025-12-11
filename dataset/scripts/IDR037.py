import os
import csv
import json
import re

from pathlib import Path

csv_path = '/data0/syhong/BioDiff/dataset/4_annotation_files/idr0037-screenA-annotation.csv'
base_dir = '/lab310/enhance/BioDiffuse_Base/IDR_raw/idr0037-vigilante-hipsci/images'

metadata = []
with open(csv_path, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        metadata.append(row)

experiment_index = {}
for row in metadata:
    comment = row['Comment [Operetta Plate Name]']
    experiment_index.setdefault(comment, []).append(row)

channel_map = {
    'ch1': 'DNA',
    'ch2': 'edu',
    'ch3': 'plasma'
}

output_path = '/data0/syhong/BioDiff/dataset/1_foundation_model/idr037.jsonl'
results = []

for root, dirs, files in os.walk(base_dir):
    if not root.endswith('/Images'):
        continue

    match = re.search(r'(.+?)__', Path(root).parts[-2])
    if not match:
        continue

    folder_prefix = match.group(1)
    matched_rows = []
    for key, rows in experiment_index.items():
        if folder_prefix in key:
            matched_rows.extend(rows)

    if not matched_rows:
        continue

    for file in files:
        if not file.endswith('.tiff'):
            continue

        match = re.match(r'r(\d{2})c(\d{2})f\d+p\d+-ch(\d)', file)
        if not match:
            continue

        row_num, col_num, ch_num = match.groups()
        ch_key = f'ch{ch_num}'

        if ch_key not in channel_map:
            continue

        cell_type = None
        for row in matched_rows:
            if row['Plate Row'].zfill(2) == row_num and row['Plate Column'].zfill(2) == col_num:
                cell_type = row['Characteristics [Cell Line]']
                break

        if not cell_type:
            continue

        prompt = f"{channel_map[ch_key]} of {cell_type}"
        full_path = os.path.join(root, file)

        results.append({
            "file_name": full_path,
            "text": prompt
        })

with open(output_path, 'w', encoding='utf-8') as f:
    for item in results:
        json.dump(item, f, ensure_ascii=False)
        f.write('\n')

print(f"{len(results)} files has been handled, result has been saved in {output_path}")
