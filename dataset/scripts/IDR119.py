import os
import json

base_dir = '/lab310/enhance/BioDiffuse_Base/IDR_raw/idr0119-gross-cellresponse'
output_jsonl_file = 'dataset/train/idr119.jsonl'

folder_to_text = {
    "21": "41",
    "AU": "39" 
}

def is_excluded(file_name):
    return "P_" in file_name

def get_text_from_parent_folders(file_path, folder_to_text):
    folder_path = os.path.dirname(file_path)
    
    while folder_path != base_dir:
        parent_folder_name = os.path.basename(folder_path)
        
        for folder_prefix, text_value in folder_to_text.items():
            if parent_folder_name.startswith(folder_prefix):
                return text_value
        
        folder_path = os.path.dirname(folder_path)
    
    return None

def generate_jsonl(base_dir, folder_to_text, output_jsonl_file):
    with open(output_jsonl_file, 'w', encoding='utf-8') as jsonl_file:
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file.endswith('.tif'):
                    if is_excluded(file):
                        continue
                    
                    file_path = os.path.join(root, file)
                    text_value = get_text_from_parent_folders(file_path, folder_to_text)
                    
                    if text_value:
                        data = {
                            'file_name': file_path,
                            'text': text_value
                        }
                        jsonl_file.write(json.dumps(data) + '\n')

if __name__ == "__main__":
    generate_jsonl(base_dir, folder_to_text, output_jsonl_file)

    print(f"JSONL file saved to {output_jsonl_file}")