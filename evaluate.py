import tensorflow as tf
from pathlib import Path

IMG_SIZE = (160, 160)
DATA_DIR = Path("data")
OUT_DIR = Path("outputs")
EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

model = tf.keras.models.load_model(OUT_DIR / "best_waste_cnn.keras")

test_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "test", image_size=IMG_SIZE, batch_size=32, shuffle=False)

loss, acc = model.evaluate(test_ds, verbose=0)

print("\n===== RESULTS FOR PPT / REPORT =====")
print(f"Test accuracy: {acc * 100:.2f}%")
print(f"Test loss: {loss:.4f}")
print("Classes:", test_ds.class_names)
for split in ["train", "validation", "test"]:
    n = sum(1 for f in (DATA_DIR / split).rglob("*") if f.suffix.lower() in EXT)
    print(f"{split} images: {n}")