import os
import json
from tqdm import tqdm

def generate_jsonl(data_root, output_path):
    image_dir = os.path.join(data_root, 'images')
    mask_dir = os.path.join(data_root, 'masks')
    
    if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
        raise ValueError("images或masks子目录不存在")
    
    image_files = [f for f in os.listdir(image_dir) 
                  if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    mask_files = [f for f in os.listdir(mask_dir) 
                 if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    
    image_dict = {os.path.splitext(f)[0]: f for f in image_files}
    mask_dict = {os.path.splitext(f)[0]: f for f in mask_files}

    common_keys = set(image_dict.keys()) & set(mask_dict.keys())
    
    with open(output_path, 'w') as out_file:
        for key in tqdm(common_keys, desc="Processing files"):
            image_path = os.path.abspath(os.path.join(image_dir, image_dict[key]))
            mask_path = os.path.abspath(os.path.join(mask_dir, mask_dict[key]))
            
            record = {
                "image": image_path,
                "conditioning_image": mask_path,
                "text": ""
            }
            
            out_file.write(json.dumps(record) + '\n')

if __name__ == "__main__":
    data_directory = "/data2/syhong/2018 Data Science Bowl process"
    output_jsonl = "/data0/syhong/BioDiff/dataset/2_down_stream_task/Data_Bowl.jsonl"

    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    
    print(f"start hadnle: {data_directory}")
    try:
        generate_jsonl(data_directory, output_jsonl)
        print(f"result is been saved to : {output_jsonl}")
    except Exception as e:
        print(f"ERROR: {e}")