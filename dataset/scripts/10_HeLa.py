import os
import json

# 设定特定的匹配规则
text_mapping = {
    'dna': "1",
    "dap": "1",
    'prot': {
        'erdak': "2",
        'mc151': "3",
        'h4b4': "4",
        'gpp130': "5",
        'giant': "6",
        'tfr': "7",
        'tubul': "8",
        'nucle': "9",
        'phal': "10"
    }
}

def get_text_from_path(folder_path):
    """
    根据文件夹路径来决定 text 字段的值
    """
    folder_parts = os.path.normpath(folder_path).split(os.sep)[-2:]
    folder_parts = os.path.join(folder_parts[0], folder_parts[1])
    if 'dap' in folder_parts.lower() or "dna" in folder_parts.lower():
        return "1"
    elif 'prot' in folder_parts.lower():
        for key, value in text_mapping['prot'].items():
            if key in folder_path.lower():
                return value
    return "0"  # 默认值，若没有匹配项

def process_images(root_dir, jsonl_file_path):
    """
    遍历文件夹及子文件夹，查找 dna 和 prot 文件夹下的 png 图像，生成 jsonl 文件
    """
    # 创建 jsonl 文件
    with open(jsonl_file_path, 'w') as jsonl_file:
        # 遍历根目录及其所有子文件夹
        for root, dirs, files in os.walk(root_dir):
            # 检查是否在 dna 或 prot 文件夹内
            if 'dna' in dirs or 'prot' in root:
                # 遍历所有文件，处理 png 文件
                for file_name in files:
                    if file_name.lower().endswith('.png'):
                        # 获取文件的完整路径
                        full_path = os.path.join(root, file_name)
                        
                        # 根据文件夹路径设置 text 字段
                        text_value = get_text_from_path(root)

                        # 创建 jsonl 文件的每一行
                        record = {
                            "file_name": full_path,
                            "text": text_value
                        }
                        
                        # 写入 jsonl 文件
                        jsonl_file.write(json.dumps(record) + '\n')


def main():
    # 设置根目录路径
    root_directory = "/lab310/enhance/BioDiffuse_Base/HeLa10Class2DImages_16bit_dna_protein_png"
    
    # 设置输出的 jsonl 文件路径
    jsonl_file_path = "/data0/syhong/BioDiff/dataset/train/HeLa10Class.jsonl"  # 请替换为您实际的输出路径
    
    # 处理文件夹并生成 jsonl 文件
    process_images(root_directory, jsonl_file_path)

if __name__ == '__main__':
    main()
