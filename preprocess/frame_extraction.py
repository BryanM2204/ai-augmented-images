import cv2
import os

def extract_uniform_frames(src_folders, output_folder, frames_per_vid):
    """
    Iterates through multiple source folders and extracts a fixed number 
    of frames from every video found.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for folder in src_folders:
        video_files = [f for f in os.listdir(folder) if f.endswith(('.mp4', '.avi'))]
        
        for video_file in video_files:
            video_path = os.path.join(folder, video_file)
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Temporal Uniform Sampling Logic
            interval = total_frames // (frames_per_vid + 1)
            if interval == 0: interval = 1

            for i in range(frames_per_vid):
                frame_idx = (i + 1) * interval
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    # Identifier format: FolderName_VideoName_FrameIdx.jpg
                    out_name = f"{os.path.basename(folder)}_{video_file.split('.')[0]}_f{frame_idx}.jpg"
                    cv2.imwrite(os.path.join(output_folder, out_name), frame)
                else:
                    break
            
            cap.release()
    print(f"Extraction complete for {output_folder}")

# Class 0: Extracting ~15,000 frames from Real sources
extract_uniform_frames(['/home/bam20007/ai-augmented-images/data/celeb-df-v2/Celeb-real', '/home/bam20007/ai-augmented-images/data/celeb-df-v2/YouTube-real'], 'dataset/class_0', 17)

# Class 1: Extracting ~30,000 frames from Deepfake sources
extract_uniform_frames(['/home/bam20007/ai-augmented-images/data/celeb-df-v2/Celeb-synthesis'], 'dataset/class_1', 6)