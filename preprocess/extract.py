import json
import os
import shutil

def top_off_class_0(defactify_test_path, target_dir, count_needed=1387):
    # Ensure paths are absolute
    defactify_test_path = os.path.expanduser(defactify_test_path)
    target_dir = os.path.expanduser(target_dir)
    
    jsonl_file = os.path.join(defactify_test_path, 'labels.jsonl')
    
    if not os.path.exists(jsonl_file):
        print(f"Error: Could not find {jsonl_file}")
        return

    print(f"Pulling {count_needed} real images from Defactify test set...")

    copied_count = 0
    with open(jsonl_file, 'r') as f:
        for line in f:
            if copied_count >= count_needed:
                break
                
            try:
                data = json.loads(line)
                # Ensure we are only grabbing 'real' images (label_a == 0)
                if data['label_a'] == 0:
                    filename = data['filename']
                    source_path = os.path.join(defactify_test_path, filename)
                    
                    if os.path.exists(source_path):
                        # Prefix with 'test_' to track origin and prevent collisions
                        new_name = f"test_{filename}"
                        dest_path = os.path.join(target_dir, new_name)
                        
                        # Use copy2 to preserve metadata or os.link for speed
                        shutil.copy2(source_path, dest_path)
                        copied_count += 1
            except Exception as e:
                continue

    print(f"Done! Added {copied_count} images to {target_dir}")

# --- Execution ---
test_path = '/home/bam20007/ai-augmented-images/data/defactify/test'
class_0_path = '/home/bam20007/ai-augmented-images/data/raw_data/class_0'

top_off_class_0(test_path, class_0_path)