import os
import json
import random
from glob import glob

# 配置路径
jsonl_dir = "/data0/syhong/BioDiff/dataset/5_HPA_split/train_raw"
generated_root = "/data4/syhong_temp/HPA_generated_re"
output_base = "/data0/syhong/BioDiff/dataset/5_HPA_split/HAP2"

output_dirs = {
    50: os.path.join(output_base, "train_aug50"),
    100: os.path.join(output_base, "train_aug100"),
    1000: os.path.join(output_base, "train_aug1000")
}

for path in output_dirs.values():
    os.makedirs(path, exist_ok=True)

generated_subfolders = {os.path.basename(folder).lower(): folder
                        for folder in glob(os.path.join(generated_root, "*"))
                        if os.path.isdir(folder)}

for jsonl_path in glob(os.path.join(jsonl_dir, "HPA_*.jsonl")):
    base_name = os.path.splitext(os.path.basename(jsonl_path))[0]  # HPA_xxx
    class_name = base_name.replace("HPA_", "").lower()  # xxx

    if class_name not in generated_subfolders:
        print(f"⚠️ 未找到匹配的生成图像文件夹: {class_name}")
        continue

    gen_folder = generated_subfolders[class_name]
    gen_images = sorted(glob(os.path.join(gen_folder, "*.tif")))

    if len(gen_images) < 1000:
        print(f"⚠️ 生成图像不足1000张：{class_name}")
        continue

    selected_1000 = random.sample(gen_images, 1000)
    selected_100 = random.sample(selected_1000, 100)
    selected_50 = random.sample(selected_100, 50)

    split_map = {
        50: selected_50,
        100: selected_100,
        1000: selected_1000,
    }

    with open(jsonl_path, "r") as f:
        original_lines = [json.loads(line.strip()) for line in f if line.strip()]

    for n, selected_imgs in split_map.items():
        augmented_lines = original_lines.copy()

        for img_path in selected_imgs:
            entry = {
                "file_name": img_path,
                "text": f"{class_name} (generated)"
            }
            augmented_lines.append(entry)

        output_path = os.path.join(output_dirs[n], os.path.basename(jsonl_path))
        with open(output_path, "w") as f_out:
            for item in augmented_lines:
                f_out.write(json.dumps(item) + "\n")

        print(f"{output_path}(total: {len(augmented_lines)})")
