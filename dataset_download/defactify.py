import os
from datasets import load_dataset

def download_and_setup_dataset():
    print("Starting download for Defactify_Image_Dataset:")
    
    # Define the local data directory
    data_dir = "./data/defactify"
    
    try:
        # Create the directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        print(f"\nUsing data directory: {data_dir}")
        
        # Download the dataset to the local directory (approx 7.5GB)
        dataset = load_dataset("Rajarshi-Roy-research/Defactify_Image_Dataset",cache_dir=data_dir)

        # Strip out Task B (Label_B) since that is irrelevant for our project
        dataset = dataset.remove_columns(["Label_B", "Caption"])

        print("\nDownload Complete!")
        print(f"\nStructure: {dataset}")        

        return dataset
    
    except KeyboardInterrupt:
        print("\nDownload interrupted by user.")
        raise
    
    except Exception as e:
        print(f"\nError downloading dataset: {str(e)}")
        raise

if __name__ == "__main__":
    ds = download_and_setup_dataset()