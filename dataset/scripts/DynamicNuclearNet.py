import os
import json
from pathlib import Path

category_map = {
    "HeLa": "nuclei of hela",
    "PC-3": "nuclei of PC-3",
    "3T3": "nuclei of 3T3",
    "HEK293": "nuclei of HEK293",
    "RPE": "nuclei of RPE",
    "RAW264": "nuclei of RAW264"
}

def process_directory(base_dir, output_jsonl):
    with open(output_jsonl, 'w') as f:
        for subdir in os.listdir(base_dir):
            subdir_path = os.path.join(base_dir, subdir)
            
            if os.path.isdir(subdir_path) and subdir in category_map:
                label = category_map[subdir]
                for file_name in os.listdir(subdir_path):
                    if file_name.endswith('.tif'):
                        image_path = os.path.join(subdir_path, file_name)
                        data = {
                            "file_name": image_path,
                            "text": label
                        }
                        
                        f.write(json.dumps(data) + '\n')
                        print(f"Processed: {image_path}")

if __name__ == "__main__":
    base_dir = '/lab310/enhance/BioDiffuse_Base/DynamicNuclearNet-tracking-v1_0/train/X'
    output_jsonl = '/data0/syhong/BioDiff/dataset/train/DynNucNet-tra.jsonl'
    
    # 调用函数处理目录
    process_directory(base_dir, output_jsonl)
    print(f"JSONL file saved at: {output_jsonl}")

