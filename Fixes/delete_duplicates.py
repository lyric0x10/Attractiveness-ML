import os
import pandas as pd
from PIL import Image
import imagehash

# Set your paths
IMG_DIR = "Model/Images"
CSV_PATH = "Model/Scores.csv"

# Supported image formats
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

# Dictionary to hold hashes and their corresponding file paths
hash_map = {}

def is_image_file(filename):
    return os.path.splitext(filename)[1].lower() in EXTENSIONS

def main():
    # Load scores CSV
    df = pd.read_csv(CSV_PATH, header=None, names=["File", "Score"])

    deleted_files = []

    files = [f for f in os.listdir(IMG_DIR) if is_image_file(f)]
    for filename in files:
        filepath = os.path.join(IMG_DIR, filename)
        try:
            with Image.open(filepath) as img:
                img_hash = imagehash.phash(img)
        except Exception as e:
            print(f"Failed to process {filename}: {e}")
            continue

        # Check if a similar hash already exists
        duplicate_found = False
        for existing_hash in hash_map:
            if img_hash - existing_hash <= 5:
                print(f"Duplicate found: {filename} is similar to {os.path.basename(hash_map[existing_hash])}")
                os.remove(filepath)
                deleted_files.append(filename)
                duplicate_found = True
                break

        if not duplicate_found:
            hash_map[img_hash] = filepath

    # Remove rows for deleted files
    df = df[~df["File"].isin(deleted_files)]
    df.to_csv(CSV_PATH, header=False, index=False)

    print("Done. Remaining images:", len(hash_map))

if __name__ == "__main__":
    main()
