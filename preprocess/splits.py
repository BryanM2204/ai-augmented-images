import os
import random
import shutil
from collections import defaultdict

def get_video_id(filename):
    """
    Extracts the unique video identifier to prevent data leakage.
    Celeb/YouTube: Celeb-real_id0_0000_f130.jpg -> ID: Celeb-real_id0_0000
    Defactify: test_test_001791.jpg -> ID: test_test_001791.jpg (unique)
    """
    if '_f' in filename:
        # Splits at the last occurrence of '_f' to get the video name
        return filename.rsplit('_f', 1)[0]
    
    # For Defactify images, each file is a static still and can be shuffled independently
    return filename

def split_dataset(input_root, output_root, split_ratio=(0.8, 0.1, 0.1)):
    input_root = os.path.expanduser(input_root)
    output_root = os.path.expanduser(output_root)
    
    categories = ['class_0', 'class_1', 'class_2']
    
    for cat in categories:
        in_dir = os.path.join(input_root, cat)
        if not os.path.exists(in_dir):
            print(f"Warning: {in_dir} not found. Skipping.")
            continue

        # 1. Group files by Video ID
        # Video frames end up in a list; static images end up in a list of size 1
        groups = defaultdict(list)
        for f in os.listdir(in_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                video_id = get_video_id(f)
                groups[video_id].append(f)

        # 2. Shuffle the unique IDs
        unique_ids = list(groups.keys())
        random.seed(42)  # Critical for reproducibility
        random.shuffle(unique_ids)

        # 3. Calculate split indices based on the number of GROUPS (videos), not images
        total_groups = len(unique_ids)
        train_end = int(total_groups * split_ratio[0])
        val_end = train_end + int(total_groups * split_ratio[1])

        split_assignment = {
            'train': unique_ids[:train_end],
            'val': unique_ids[train_end:val_end],
            'test': unique_ids[val_end:]
        }

        # 4. Move the files
        for split_name, ids in split_assignment.items():
            dest_dir = os.path.join(output_root, split_name, cat)
            os.makedirs(dest_dir, exist_ok=True)
            
            print(f"Moving {len(ids)} video/image groups to {split_name}/{cat}...")
            for vid_id in ids:
                for filename in groups[vid_id]:
                    src = os.path.join(in_dir, filename)
                    dst = os.path.join(dest_dir, filename)
                    shutil.move(src, dst)

    print("\nDataset split complete!")

# --- Execution ---
# Note: Ensure the input_data path matches where your padded/resized images live
input_data = '~/ai-augmented-images/preprocessed_data_v2'
output_data = '~/ai-augmented-images/data/final_dataset_v2'

split_dataset(input_data, output_data)