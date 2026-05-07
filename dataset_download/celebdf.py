import os
import kaggle

def download_celeb_df():
    # The dataset identifier from the Kaggle URL
    dataset_path = 'reubensuju/celeb-df-v2'
    
    # Define the target directory for your data pipeline
    # Ensure the drive hosting this path has sufficient storage!
    download_dir = './data/celeb-df-v2'
    
    # Create the data directory if it doesn't already exist
    os.makedirs(download_dir, exist_ok=True)
    
    print(f"Initiating download for '{dataset_path}'...")
    print(f"Target directory: {os.path.abspath(download_dir)}")
    
    try:
        # Download and automatically unzip the contents
        kaggle.api.dataset_download_files(
            dataset_path, 
            path=download_dir, 
            unzip=True
        )
        print("Download and extraction completed successfully!")
        
    except Exception as e:
        print(f"An error occurred during the download: {e}")

if __name__ == "__main__":
    download_celeb_df()