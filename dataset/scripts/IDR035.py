import os
import csv
import json

images_dir = '/lab310/enhance/BioDiffuse_Base/IDR_raw/idr0035-caie-drugresponse/images'
csv_file = '/lab310/enhance/BioDiffuse_Base/IDR_raw/idr0035-caie-drugresponse/metadata/BBBC021_v1_image.csv'
output_jsonl_file = 'dataset/train/idr035.jsonl'

def load_csv(csv_file):
    label_map = {}
    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['Image_FileName_DAPI']:
                label_map[row['Image_FileName_DAPI']] = "28"
            if row['Image_FileName_Tubulin']:
                label_map[row['Image_FileName_Tubulin']] = "29"
            if row['Image_FileName_Actin']:
                label_map[row['Image_FileName_Actin']] = "30"
    return label_map

def generate_jsonl(images_dir, label_map, output_jsonl_file):
    with open(output_jsonl_file, 'w', encoding='utf-8') as jsonl_file:
        for root, dirs, files in os.walk(images_dir):
            for file in files:
                if file.endswith('.tif'):
                    file_path = os.path.join(root, file)
                    text = label_map.get(file, None)
                    if text is not None:
                        data = {
                            'file_name': file_path,
                            'text': str(text)
                        }
                        jsonl_file.write(json.dumps(data) + '\n')
                    else:
                        print(f"Warning: No label found for {file_path}")

if __name__ == "__main__":
    label_map = load_csv(csv_file)
    
    generate_jsonl(images_dir, label_map, output_jsonl_file)
    print(f"JSONL file saved to {output_jsonl_file}")

