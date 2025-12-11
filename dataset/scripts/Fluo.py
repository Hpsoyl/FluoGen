import os
import glob
import json

base_folders = [
    '/lab310/enhance/BioDiffuse_Base/Fluo-C2DL-Huh7',
    '/lab310/enhance/BioDiffuse_Base/Fluo-C2DL-MSC',
    '/lab310/enhance/BioDiffuse_Base/Fluo-N2DH-GOWT1',
    '/lab310/enhance/BioDiffuse_Base/Fluo-N2DH-SIM+',
    '/lab310/enhance/BioDiffuse_Base/Fluo-N2DL-HeLa'
]

labels = {
    '/lab310/enhance/BioDiffuse_Base/Fluo-C2DL-Huh7': '12',
    '/lab310/enhance/BioDiffuse_Base/Fluo-C2DL-MSC': '13',
    '/lab310/enhance/BioDiffuse_Base/Fluo-N2DH-GOWT1': '14',
    '/lab310/enhance/BioDiffuse_Base/Fluo-N2DH-SIM+': '15',
    '/lab310/enhance/BioDiffuse_Base/Fluo-N2DL-HeLa': '16'
}


def get_image_paths(base_folders, labels):
    result = []
   
    for folder in base_folders:
        label = labels.get(folder, None)
        if not label:
            continue

        for subfolder in ['01', '02']:
            subfolder_path = os.path.join(folder, subfolder)
            if os.path.isdir(subfolder_path):
                tif_files = glob.glob(os.path.join(subfolder_path, '*.tif'))
                for tif_file in tif_files:
                    result.append({
                        'file_name': tif_file,
                        'text': label
                    })
    return result

def save_to_jsonl(result, output_file='output.jsonl'):
    with open(output_file, 'w') as f:
        for item in result:
            json.dump(item, f)
            f.write('\n')

if __name__ == "__main__":
    image_paths = get_image_paths(base_folders, labels)
    
    save_to_jsonl(image_paths, 'dataset/train/Fluo.jsonl')
    print(f"{len(image_paths)} files has been saved")
