
from pathlib import Path
import random
import shutil

#source folder
source = Path("asl_dataset")
#filtered data folder
destination = Path("dataset")

random.seed(42)

train_ratio = 0.70
val_ratio = 0.15

extensions = {".jpg", ".jpeg", ".png", ".webp"}

for class_folder in sorted(source.iterdir()):
    if not class_folder.is_dir():
        continue

    images = [
        p for p in class_folder.iterdir()
        if p.suffix.lower() in extensions
    ]

    random.shuffle(images)

    n = len(images)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:]
    }

    for split_name, files in splits.items():
        output = destination / split_name / class_folder.name
        output.mkdir(parents=True, exist_ok=True)

        for file in files:
            shutil.copy2(file, output / file.name)

    print(f"{class_folder.name}: {n} images split")

print("Dataset split completed!")
