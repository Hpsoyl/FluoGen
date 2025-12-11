import os
import json

folder_path = '/lab310/enhance/BioDiffuse_Base/FMD/test_mix/raw'
text_rules = {
    'MICE': 'mouse brain tissues',
    'FISH': 'zebrafish embryos',
    'BPAE_B': 'nucleus of BPAE',
    'BPAE_R': 'mitochondria of BPAE',
    'BPAE_G': 'F-actin of BPAE'
}
output_file = 'dataset/validation_image/FMD_test_mix.jsonl'

with open(output_file, 'w') as jsonl_file:
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.png'):
                image_path = os.path.join(root, file)
                
                folder_name = os.path.basename(image_path)
                text_value = next((text for key, text in text_rules.items() if key in folder_name), 'unknown')
            
                json_obj = {
                    "conditioning_image": image_path,
                    "text": text_value
                }
                jsonl_file.write(json.dumps(json_obj) + '\n')

print(f'JSONL file has been saved to {output_file}')
