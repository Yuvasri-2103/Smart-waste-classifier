import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from pathlib import Path

IMG_SIZE = (160, 160)
OUT_DIR = Path("outputs")

# Bin type and disposal tip per class
INFO = {
    "cardboard": ("Recyclable", "Flatten the box and keep it dry before recycling."),
    "glass": ("Recyclable", "Rinse and place in the glass bin. Do not mix with ceramics."),
    "metal": ("Recyclable", "Rinse cans and tins, then place in the metal recycling bin."),
    "paper": ("Recyclable", "Keep it clean and dry. Greasy or wet paper is not recyclable."),
    "plastic": ("Recyclable", "Rinse and check the recycling number before binning."),
    "trash": ("Non-recyclable", "Dispose in the general waste bin."),
}

st.set_page_config(page_title="Smart Waste Classifier", page_icon="♻️")


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(OUT_DIR / "best_waste_cnn.keras")
    names = (OUT_DIR / "class_names.txt").read_text(encoding="utf-8").split("\n")
    return model, names


model, class_names = load_model()

st.title("♻️ Smart Waste Classifier")
st.write("Upload a waste image and the model will predict its category.")

file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])

if file is not None:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Uploaded image", use_container_width=True)

    arr = np.array(img.resize(IMG_SIZE, Image.BILINEAR), dtype="float32")
    probs = model.predict(np.expand_dims(arr, 0), verbose=0)[0]
    best = int(np.argmax(probs))
    label = class_names[best]
    conf = float(probs[best])

    st.subheader(f"Prediction: {label.title()}")
    st.progress(conf, text=f"Confidence: {conf * 100:.1f}%")

    kind, tip = INFO.get(label, ("", ""))
    if kind:
        st.info(f"**{kind}** - {tip}")

    if conf < 0.6:
        st.warning("Low confidence. Try a clearer photo with a plain background.")

    st.bar_chart({name: float(p) for name, p in zip(class_names, probs)})