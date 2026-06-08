#Used to check if YAML file is loading fine
from pathlib import Path

root = Path(
    r"C:\Users\Aniruddh\OneDrive\Documents\PotholeProject\PotholeProject\datasets"
)

for split in ["train", "valid"]:

    images = len(list((root / split / "images").glob("*.*")))
    labels = len(list((root / split / "labels").glob("*.txt")))

    print(f"{split}:")
    print(f"  images = {images}")
    print(f"  labels = {labels}")