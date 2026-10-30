
from pathlib import Path

# Your existing Food-101 dataset location
DATASET_PATH = Path.home() / "Downloads" / "archive (4)" / "food-101" / "food-101" / "meta" / "images"

print("=== FOODVISION AI DATASET CHECK ===")
print("Dataset path:", DATASET_PATH)
print("Dataset exists:", DATASET_PATH.exists())

if DATASET_PATH.exists():
    classes = [p for p in DATASET_PATH.iterdir() if p.is_dir()]
    print("Food folders found:", len(classes))
    print("Sample folders:", [p.name for p in classes[:10]])
else:
    print("Folder not found. Check the exact Downloads folder name.")
