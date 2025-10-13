import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# ----------------------------
# 1. PAGE SETUP
# ----------------------------
st.title("🍲 Smart Recipe Recommender - Version 2.0")
st.write("Welcome, Mohanad! 👋")
st.write("Now powered by AI — upload an image to identify the food item!")

# ----------------------------
# 2. LOAD MODEL + LABELS (cached)
# ----------------------------
@st.cache_resource
def load_model_and_labels():
    model = tf.keras.models.load_model("fruit_model.h5")
    with open("class_indices.json") as f:
        class_indices = json.load(f)
    labels = {v: k for k, v in class_indices.items()}
    return model, labels

model, labels = load_model_and_labels()

# ----------------------------
# 3. USER INPUTS
# ----------------------------
image = st.file_uploader("📸 Upload a food image", type=["jpg", "png"])
calories = st.number_input("🔥 Calories per serving", 50, 1000, 300)
servings = st.number_input("👥 Number of servings", 1, 10, 2)
recipe_type = st.selectbox("🍴 Recipe Type", ["Sweet", "Savory", "Drink", "Snack"])
protein_range = st.slider("💪 Protein (grams)", 0, 50, (5, 20))
sugar_range = st.slider("🍬 Sugar (grams)", 0, 50, (0, 15))

# ----------------------------
# 4. PREDICTION LOGIC
# ----------------------------
if st.button("🔍 Identify Food"):
    if image is not None:
        # Load and preprocess image
        img = Image.open(image).convert("RGB").resize((100, 100))
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

        # Predict
        preds = model.predict(img_array)
        class_id = int(np.argmax(preds))
        confidence = float(np.max(preds) * 100)
        predicted_label = labels[class_id]

        # Display results
        st.image(image, caption=f"📷 Uploaded Image — Detected: {predicted_label}", use_container_width=True)
        st.success(f"✅ Predicted: **{predicted_label}** ({confidence:.2f}% confidence)")

        # (Future) connect to recipe recommendations here
        st.info(f"Next: Filter recipes for **{predicted_label}**, {recipe_type} type, {calories} kcal/serving.")

    else:
        st.warning("⚠️ Please upload an image before running prediction.")
