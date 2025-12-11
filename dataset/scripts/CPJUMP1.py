import os
import json

base_dir = '/lab310/enhance/BioDiffuse_Base/CPJUMP1/images'
output_jsonl_file = 'dataset/train/CPJUMP1.jsonl'
file_keywords_to_text = {
    "ch1": "55",
    "ch2": "56",
    "ch3": "57",
    "ch4": "58",
    "ch5": "59"
}

def generate_jsonl(base_dir, file_keywords_to_text, output_jsonl_file):
    with open(output_jsonl_file, 'w', encoding='utf-8') as jsonl_file:
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file.endswith('.tiff'):
                    for keyword, text_value in file_keywords_to_text.items():
                        if keyword.lower() in file.lower():
                            file_path = os.path.join(root, file)
                            data = {
                                'file_name': file_path,
                                'text': text_value
                            }
                            jsonl_file.write(json.dumps(data) + '\n')
                            break

if __name__ == "__main__":
    generate_jsonl(base_dir, file_keywords_to_text, output_jsonl_file)

    print(f"JSONL file saved to {output_jsonl_file}")
