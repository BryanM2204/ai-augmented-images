import cv2
import os
import glob

def extract_specific_frames(video_dir, output_dir, frames_per_video, total_target_frames):
    """
    Extracts a specific number of evenly spaced frames per video until a target limit is reached.
    """
    # Grab all mp4 files in the target directory
    video_files = glob.glob(os.path.join(video_dir, '*.mp4'))
    num_videos = len(video_files)
    
    if num_videos == 0:
        print(f"No videos found in {video_dir}.")
        return

    print(f"Found {num_videos} videos. Extracting ~{frames_per_video} frames per video.")
    os.makedirs(output_dir, exist_ok=True)
    
    frame_count = 0

    for video_path in video_files:
        vid_name = os.path.basename(video_path).split('.')[0]
        cap = cv2.VideoCapture(video_path)
        
        # Get total frames to calculate even spacing
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Guard against exceptionally short videos
        actual_frames_to_extract = min(frames_per_video, total_frames)
        if actual_frames_to_extract == 0:
            cap.release()
            continue
            
        interval = max(1, total_frames // actual_frames_to_extract)
        
        extracted_from_this_vid = 0
        current_frame = 0
        
        while cap.isOpened() and extracted_from_this_vid < actual_frames_to_extract:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            
            if not ret:
                break
                
            # Save the image
            output_filename = os.path.join(output_dir, f"{vid_name}_frame_{current_frame}.jpg")
            cv2.imwrite(output_filename, frame)
            
            extracted_from_this_vid += 1
            frame_count += 1
            current_frame += interval
            
            # Hard stop condition
            if frame_count >= total_target_frames:
                cap.release()
                print(f"\nTarget of {total_target_frames} frames reached!")
                return

        cap.release()
        print(f"Processed {vid_name} | Total Extracted: {frame_count}", end='\r')

    print(f"\nExtraction complete. Total frames saved: {frame_count}")

# ==========================================
# Execution
# ==========================================
if __name__ == "__main__":
    # 1. Extract 15,000 Authentic Images (26 per video)
    extract_specific_frames(
        video_dir='./data/celeb-df-v2-videos/Celeb-real', 
        output_dir='./data/extracted/class_0_authentic', 
        frames_per_video=26, 
        total_target_frames=15000
    )

    # 2. Extract 15,000 Deepfake Images (6 per video)
    extract_specific_frames(
        video_dir='./data/celeb-df-v2-videos/Celeb-synthesis', 
        output_dir='./data/extracted/class_1_deepfake', 
        frames_per_video=6, 
        total_target_frames=15000
    )