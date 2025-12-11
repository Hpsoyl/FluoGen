import os
import json

def traverse_images_and_write_jsonl(root_dir, jsonl_file_path):
    with open(jsonl_file_path, 'w') as jsonl_file:
        for root, dirs, files in os.walk(root_dir):
            if 'images' in root.lower():
                for file in files:
                    if file.endswith('.png'):
                        file_path = os.path.join(root, file)
                        data = {"file_name": file_path, "text": "11"}
                        jsonl_file.write(json.dumps(data) + '\n')

root_directory = '/lab310/enhance/BioDiffuse_Base/2018 Data Science Bowl select'
jsonl_output_path = 'dataset/train/2018_Data_Science_Bowl.jsonl'

traverse_images_and_write_jsonl(root_directory, jsonl_output_path)
print(f".png file path has been writen into {jsonl_output_path}")
