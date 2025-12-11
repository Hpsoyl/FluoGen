import os
import json

sample = "CCPs"
def generate_jsonl(directory, output_file):
    entries = []

    for root, dirs, files in os.walk(directory):
        if 'training_gt' not in root:
            continue
        for file in files:
            if file.endswith('.tif'):
                image_path = os.path.join(root, file)
                conditioning_path = image_path.replace('training_gt', 'training_9', 1)
                entry = {
                    "image": image_path,
                    "conditioning_image": conditioning_path,
                    "text": f"{sample} of COS-7"
                }
                entries.append(entry)

        with open(output_file, 'w') as f:
            for entry in entries:
                json_entry = json.dumps(entry)
                f.write(json_entry + '\n')

directory = f'/data4/syhong_temp/data/BioSR_Reduce_512/{sample}'
output_file = f'/data0/syhong/BioDiff/dataset/2_down_stream_task/BioSR_{sample}_Reduce.jsonl'

generate_jsonl(directory, output_file)
