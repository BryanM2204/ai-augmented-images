import json
import os
import shutil

def sort_defactify_train_val(base_path, output_path):
    # Ensure paths are absolute (expands ~)
    base_path = os.path.expanduser(base_path)
    output_path = os.path.expanduser(output_path)
    
    # We want these to go into a 'raw' directory before the final padding/resize
    raw_dir = os.path.join(output_path, 'raw_data')
    class_0_dir = os.path.join(raw_dir, 'class_0')
    class_2_dir = os.path.join(raw_dir, 'class_2')
    
    os.makedirs(class_0_dir, exist_ok=True)
    os.makedirs(class_2_dir, exist_ok=True)

    splits = ['train', 'validation']

    for split in splits:
        split_dir = os.path.join(base_path, split)
        jsonl_file = os.path.join(split_dir, 'labels.jsonl') 
        
        if not os.path.exists(jsonl_file):
            print(f"Warning: Could not find {jsonl_file}. Skipping {split}.")
            continue

        print(f"Sorting images from: {split}...")
        
        with open(jsonl_file, 'r') as f:
            count = 0
            for line in f:
                try:
                    data = json.loads(line)
                    image_name = data['filename']
                    label = data['label_a'] 
                    
                    source_img_path = os.path.join(split_dir, image_name)
                    
                    if os.path.exists(source_img_path):
                        # Unique name to prevent overlap between train/val sets
                        new_name = f"{split}_{image_name}"
                        
                        if label == 0: # Real
                            target_path = os.path.join(class_0_dir, new_name)
                            shutil.copy2(source_img_path, target_path)
                        elif label == 1: # AI-Generated
                            target_path = os.path.join(class_2_dir, new_name)
                            shutil.copy2(source_img_path, target_path)
                        
                        count += 1
                except Exception as e:
                    pass # Skip incomplete lines/errors
            
            print(f"Successfully sorted {count} images from {split}.")

# Execute the sorting
base_defactify_path = '/home/bam20007/ai-augmented-images/data/defactify'
output_destination = '~/ai-augmented-images/data/'
sort_defactify_train_val(base_defactify_path, output_destination)