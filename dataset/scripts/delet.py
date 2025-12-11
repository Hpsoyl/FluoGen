import json

input_path = "/data0/syhong/BioDiff/dataset/2_down_stream_task/FMD_denoising.jsonl"
output_path = "/data0/syhong/BioDiff/dataset/2_down_stream_task/FMD_denoising_reduce.jsonl"

seen_dirs = set()
filtered_lines = []

with open(input_path, "r") as f:
    for line in f:
        entry = json.loads(line.strip())
        cond_img = entry["conditioning_image"]

        dir_path = "/".join(cond_img.split("/")[:-1])

        if dir_path not in seen_dirs:
            seen_dirs.add(dir_path)
            filtered_lines.append(entry)

with open(output_path, "w") as f:
    for entry in filtered_lines:
        f.write(json.dumps(entry) + "\n")

print(f"result has been saved in  {output_path}, {len(filtered_lines)} has been saved")
