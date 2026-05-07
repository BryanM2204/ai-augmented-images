import os
import argparse
import csv
import json
from datasets import load_dataset
from datasets import Image


def export_images(dataset, output_dir, max_images_per_split=None):
    """Write image bytes from HF Arrow cache to actual image files on disk."""
    os.makedirs(output_dir, exist_ok=True)
    total_written = 0

    for split_name, split_ds in dataset.items():
        split_out = os.path.join(output_dir, split_name)
        os.makedirs(split_out, exist_ok=True)

        raw_split = split_ds.cast_column("Image", Image(decode=False))
        split_count = 0
        split_metadata = []

        for idx, row in enumerate(raw_split):
            if max_images_per_split is not None and split_count >= max_images_per_split:
                break

            image_info = row["Image"]
            img_bytes = image_info.get("bytes") if isinstance(image_info, dict) else None
            img_name = image_info.get("path") if isinstance(image_info, dict) else None

            if not img_bytes:
                continue

            # Keep original extension when available; otherwise fallback to .jpg.
            if img_name:
                _, ext = os.path.splitext(img_name)
            else:
                ext = ".jpg"

            if not ext:
                ext = ".jpg"

            out_name = f"{split_name}_{idx:06d}{ext.lower()}"
            out_path = os.path.join(split_out, out_name)
            with open(out_path, "wb") as f:
                f.write(img_bytes)

            label_a = row.get("Label_A") if isinstance(row, dict) else None
            split_metadata.append(
                {
                    "filename": out_name,
                    "row_index": idx,
                    "label_a": label_a,
                    "label_a_name": "real" if label_a == 0 else "ai-generated" if label_a == 1 else None,
                }
            )

            split_count += 1
            total_written += 1

        # Save label mapping for each exported split image.
        metadata_csv_path = os.path.join(split_out, "labels.csv")
        with open(metadata_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["filename", "row_index", "label_a", "label_a_name"]
            )
            writer.writeheader()
            writer.writerows(split_metadata)

        metadata_jsonl_path = os.path.join(split_out, "labels.jsonl")
        with open(metadata_jsonl_path, "w", encoding="utf-8") as f:
            for row in split_metadata:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")

        print(
            f"Exported {split_count} images for split '{split_name}' to {split_out} "
            f"(+ labels.csv, labels.jsonl)"
        )

    print(f"\nTotal exported images: {total_written}")


def download_and_setup_dataset(cache_dir, export_dir=None, max_images_per_split=None):
    print("Starting download for Defactify_Image_Dataset:")

    data_dir = os.path.abspath(os.path.expanduser(cache_dir))
    
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

        if export_dir:
            export_root = os.path.abspath(os.path.expanduser(export_dir))
            print(f"\nMaterializing images to: {export_root}")
            export_images(dataset, export_root, max_images_per_split=max_images_per_split)

        return dataset
    
    except KeyboardInterrupt:
        print("\nDownload interrupted by user.")
        raise
    
    except Exception as e:
        print(f"\nError downloading dataset: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Defactify dataset from Hugging Face.")
    parser.add_argument(
        "--cache-dir",
        default="./data/defactify",
        help="Directory for Hugging Face Arrow cache files.",
    )
    parser.add_argument(
        "--export-images-dir",
        default=None,
        help="If set, exports images to this directory as regular files.",
    )
    parser.add_argument(
        "--max-images-per-split",
        type=int,
        default=None,
        help="Optional limit for quick tests.",
    )
    args = parser.parse_args()

    ds = download_and_setup_dataset(
        cache_dir=args.cache_dir,
        export_dir=args.export_images_dir,
        max_images_per_split=args.max_images_per_split,
    )