import os
import json

base_path = "/lab310/enhance/BioDiffuse_Base/DynamicNuclearNet-segmentation-v1_0"
image_path = os.path.join(base_path, "train/X")
conditioning_image_path = os.path.join(base_path, "train/y")
output_jsonl_path = "dataset/2_down_stream_task/Cell_count.jsonl"

text_mapping = {
    "HeLa": "nuclei of hela",
    "PC-3": "nuclei of PC-3",
    "3T3": "nuclei of 3T3",
    "HEK293": "nuclei of HEK293",
    "RPE": "nuclei of RPE",
    "RAW264": "nuclei of RAW264",
}

with open(output_jsonl_path, "w") as f:
    for subfolder in os.listdir(image_path):
        subfolder_path = os.path.join(image_path, subfolder)
        if os.path.isdir(subfolder_path):
            for file in os.listdir(subfolder_path):
                if file.endswith(".tif"):
                    image_file_path = os.path.join(subfolder_path, file)
                    conditioning_image_file_path = os.path.join(conditioning_image_path, subfolder, file)
                    
                    if not os.path.exists(conditioning_image_file_path):
                        print(f"warning: conditioning_image file {conditioning_image_file_path} is not existed, scape")
                        continue
                    
                    text = text_mapping.get(subfolder, f"nuclei of {subfolder}")
                    json_obj = {
                        "image": image_file_path,
                        "conditioning_image": conditioning_image_file_path,
                        "text": text,
                    }
                    
                    f.write(json.dumps(json_obj) + "\n")

print(f"JSONL file has been generated: {output_jsonl_path}")