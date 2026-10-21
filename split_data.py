
import random
import shutil
from pathlib import Path

random.seed(42)

source = Path("dataset-resized")
output = Path("data")

for category in source.iterdir():
    if not category.is_dir():
        continue

    images = [
        f for f in category.iterdir()
        if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
    ]
    random.shuffle(images)

    n = len(images)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    splits = {
        "train": images[:train_end],
        "validation": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split, files in splits.items():
        dest = output / split / category.name
        dest.mkdir(parents=True, exist_ok=True)

        for file in files:
            shutil.copy2(file, dest / file.name)

    print(f"{category.name}: {n} images split")

print("Dataset split completed!")
