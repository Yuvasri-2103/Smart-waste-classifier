import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")  # save plots without blocking the terminal
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras import layers

# ---------------- Settings ----------------
IMG_SIZE = (160, 160)
BATCH_SIZE = 32
DATA_DIR = Path("data")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

EPOCHS_FROZEN = 10      # phase 1: train only the new head
EPOCHS_FINETUNE = 15    # phase 2: fine-tune top layers of MobileNetV2
UNFREEZE_LAYERS = 30


def save_fig(name):
    """Save the current plot; if the file is locked, save under a new name."""
    try:
        plt.savefig(OUT_DIR / name)
    except OSError:
        alt = OUT_DIR / ("new_" + name)
        plt.savefig(alt)
        print(f"Could not overwrite {name} (open elsewhere). Saved as {alt.name}")

# ---------------- Data ----------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "train", image_size=IMG_SIZE, batch_size=BATCH_SIZE)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "validation", image_size=IMG_SIZE, batch_size=BATCH_SIZE)
test_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "test", image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False)

class_names = train_ds.class_names
num_classes = len(class_names)
print("Waste classes:", class_names)
(OUT_DIR / "class_names.txt").write_text("\n".join(class_names), encoding="utf-8")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# ---------------- Class weights (handles imbalance) ----------------
counts = np.array([
    len([f for f in (DATA_DIR / "train" / c).iterdir() if f.is_file()])
    for c in class_names
])
total = counts.sum()
class_weight = {i: total / (num_classes * counts[i]) for i in range(num_classes)}
print("Images per class:", dict(zip(class_names, counts.tolist())))

# ---------------- Model ----------------
augment = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
], name="augmentation")

base = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
base.trainable = False

inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = augment(inputs)
x = layers.Rescaling(1.0 / 127.5, offset=-1)(x)  # MobileNetV2 expects [-1, 1]
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

# ---------------- Phase 1: train the head ----------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

ckpt_path = OUT_DIR / "best_waste_cnn.keras"

print("\n=== Phase 1: training the new head ===")
history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FROZEN,
    class_weight=class_weight,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor="val_accuracy", mode="max", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.3),
    ],
)
best_val_phase1 = max(history1.history["val_accuracy"])

# ---------------- Phase 2: fine-tune ----------------
base.trainable = True
for layer in base.layers[:-UNFREEZE_LAYERS]:
    layer.trainable = False
for layer in base.layers:  # keep BatchNorm stable
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

print("\n=== Phase 2: fine-tuning ===")
history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FROZEN + EPOCHS_FINETUNE,
    initial_epoch=len(history1.history["accuracy"]),
    class_weight=class_weight,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor="val_accuracy", mode="max", save_best_only=True,
            initial_value_threshold=best_val_phase1),
        tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.3),
    ],
)

# ---------------- Save training history (so graphs can be redrawn) ----------------
import json
(OUT_DIR / "history.json").write_text(json.dumps({
    k: [float(v) for v in history1.history[k] + history2.history[k]]
    for k in history1.history
}), encoding="utf-8")

# ---------------- Evaluate ----------------
loss, accuracy = model.evaluate(test_ds)
print(f"\nTest accuracy: {accuracy:.4f}")
print(f"Test loss: {loss:.4f}")

model.save(OUT_DIR / "waste_cnn.keras")

# ---------------- Accuracy / loss graphs ----------------
acc = history1.history["accuracy"] + history2.history["accuracy"]
val_acc = history1.history["val_accuracy"] + history2.history["val_accuracy"]
tr_loss = history1.history["loss"] + history2.history["loss"]
val_loss = history1.history["val_loss"] + history2.history["val_loss"]
split = len(history1.history["accuracy"]) - 1

plt.figure()
plt.plot(acc, label="Training")
plt.plot(val_acc, label="Validation")
plt.axvline(split, color="gray", linestyle="--", label="Fine-tuning starts")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
save_fig("accuracy.png")
plt.close()

plt.figure()
plt.plot(tr_loss, label="Training")
plt.plot(val_loss, label="Validation")
plt.axvline(split, color="gray", linestyle="--", label="Fine-tuning starts")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
save_fig("loss.png")
plt.close()

# ---------------- Confusion matrix + per-class report ----------------
y_true = np.concatenate([y.numpy() for _, y in test_ds])
y_pred = np.argmax(model.predict(test_ds), axis=1)
cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=num_classes).numpy()

fig, ax = plt.subplots(figsize=(7, 6))
ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(num_classes))
ax.set_yticks(range(num_classes))
ax.set_xticklabels(class_names, rotation=45, ha="right")
ax.set_yticklabels(class_names)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
for i in range(num_classes):
    for j in range(num_classes):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.tight_layout()
save_fig("confusion_matrix.png")
plt.close()

print("\nPer-class results:")
print(f"{'Class':<15}{'Precision':>10}{'Recall':>10}{'F1':>10}")
for i, name in enumerate(class_names):
    tp = cm[i, i]
    precision = tp / max(cm[:, i].sum(), 1)
    recall = tp / max(cm[i, :].sum(), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    print(f"{name:<15}{precision:>10.2f}{recall:>10.2f}{f1:>10.2f}")

print("\nTraining completed!")
print("Saved in outputs/: best_waste_cnn.keras, waste_cnn.keras, accuracy.png, "
      "loss.png, confusion_matrix.png, class_names.txt")