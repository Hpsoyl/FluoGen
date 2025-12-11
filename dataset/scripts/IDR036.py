import os
import json

base_dir = '/lab310/enhance/BioDiffuse_Base/IDR_raw/idr0036-gustafsdottir-cellpainting/2016-01-19-screens-bbbc022'
output_jsonl_file = 'dataset/train/idr036.jsonl'

folder_to_text = {
    "ERSyto": "32",
    "ERSytoBleed": "33",
    "Hoechst": "31",
    "Mito": "34",
    "Ph_golgi": "35"
}

def generate_jsonl(base_dir, folder_to_text, output_jsonl_file):
    with open(output_jsonl_file, 'w', encoding='utf-8') as jsonl_file:
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file.endswith('.tif'):
                    folder_name = os.path.basename(root)
                    for key, text in folder_to_text.items():
                        if folder_name.endswith(key):
                            file_path = os.path.join(root, file)
                            data = {
                                'file_name': file_path,
                                'text': text
                            }
                            jsonl_file.write(json.dumps(data) + '\n')
                            break 

if __name__ == "__main__":
    generate_jsonl(base_dir, folder_to_text, output_jsonl_file)

    print(f"JSONL file saved to {output_jsonl_file}")
