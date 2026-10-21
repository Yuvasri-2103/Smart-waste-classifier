import sys
import random
import numpy as np
import tensorflow as tf
from pathlib import Path

IMG_SIZE = (160, 160)
OUT_DIR = Path("outputs")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_FROM_FOLDER = 5  # if a folder is given, test this many random images

model = tf.keras.models.load_model(OUT_DIR / "best_waste_cnn.keras")
class_names = (OUT_DIR / "class_names.txt").read_text(encoding="utf-8").split("\n")

# Usage: python predict.py "path\to\image.jpg"   or   python predict.py "path\to\folder"
if len(sys.argv) > 1:
    target = sys.argv[1]
else:
    target = input("Enter image or folder path: ")
target = Path(target.strip().strip('"'))

if target.is_dir():
    files = [f for f in target.iterdir() if f.suffix.lower() in IMG_EXT]
    if not files:
        sys.exit("No images found in that folder.")
    files = random.sample(files, min(MAX_FROM_FOLDER, len(files)))
elif target.is_file():
    files = [target]
else:
    sys.exit(f"Path not found: {target}")


def predict(path):
    img = tf.keras.utils.load_img(path, target_size=IMG_SIZE)
    arr = np.expand_dims(tf.keras.utils.img_to_array(img), axis=0)  # model rescales internally
    probs = model.predict(arr, verbose=0)[0]
    top3 = np.argsort(probs)[::-1][:3]
    print(f"\n{path.name}")
    print(f"  Prediction: {class_names[top3[0]]} ({probs[top3[0]] * 100:.1f}%)")
    for i in top3:
        print(f"    {class_names[i]:<12} {probs[i] * 100:5.1f}%")


for f in files:
    predict(f)