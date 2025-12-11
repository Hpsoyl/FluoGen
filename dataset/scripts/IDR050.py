import os
import json

base_dir = '/lab310/enhance/BioDiffuse_Base/IDR/idr0050/20181106'
output_jsonl_file = 'dataset/train/idr050.jsonl'

folder_to_text = {
    "Actin": "36",
    "Microtubules": "37",
    "Vimentin": "38"
}

folder_to_text = {
    "Actin": "36",
    "Microtubules": "37",
    "Vimentin": "38"
}

def generate_jsonl(base_dir, folder_to_text, output_jsonl_file):
    with open(output_jsonl_file, 'w', encoding='utf-8') as jsonl_file:
        for root, dirs, files in os.walk(base_dir):
            folder_name = os.path.basename(root)
            
            if folder_name == "Cell":
                continue
            
            if folder_name in folder_to_text:
                for file in files:
                    if file.endswith('.tif') or file.endswith('.tiff'):
                        file_path = os.path.join(root, file)
                        data = {
                            'file_name': file_path,
                            'text': folder_to_text[folder_name]
                        }
                        jsonl_file.write(json.dumps(data) + '\n')

if __name__ == "__main__":
    generate_jsonl(base_dir, folder_to_text, output_jsonl_file)

    print(f"JSONL file saved to {output_jsonl_file}")