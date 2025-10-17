import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import json
import requests
import os
import re

# -----------------------------
# 1. PAGE SETUP
# -----------------------------
st.set_page_config(page_title="Smart Recipe Recommender", page_icon="🍲", layout="centered")
st.title("🍲 Smart Recipe Recommender - Graduate Project")
st.write("Upload a fruit or vegetable image and discover healthy recipes based on AI-powered recognition 🍎🥦")

# -----------------------------
# 2. LOAD MODEL + LABEL MAP
# -----------------------------
@st.cache_resource
def load_model_and_labels():
    model_path = os.path.join("models", "fruit_model.h5")
    labels_path = os.path.join("models", "class_indices.json")

    model = tf.keras.models.load_model(model_path)
    with open(labels_path, "r") as f:
        class_indices = json.load(f)
    index_to_label = {v: k for k, v in class_indices.items()}
    return model, index_to_label

model, index_to_label = load_model_and_labels()

# -----------------------------
# 3. LABEL VARIANT GENERATOR
# -----------------------------
def expand_label_variants(raw_label):
    """
    Generate multiple search variations from a model label.
    Example:
        'banana_lady_finger_1' ->
        ['banana lady finger', 'banana lady', 'banana']
    """
    label = raw_label.lower().strip()
    label = label.replace("_", " ").replace("-", " ")
    label = re.sub(r"[^a-z\s]", "", label).strip()

    words = label.split()
    if not words:
        return [raw_label]

    # Generate descending combinations (most specific → general)
    variants = []
    for i in range(len(words), 0, -1):
        phrase = " ".join(words[:i])
        if phrase not in variants:
            variants.append(phrase)
    return variants

# -----------------------------
# 4. PREDICTION FUNCTION
# -----------------------------
def predict_image(img):
    img = image.load_img(img, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    preds = model.predict(x)
    predicted_idx = np.argmax(preds)
    confidence = preds[0][predicted_idx]
    label = index_to_label[predicted_idx]
    return label, confidence

# -----------------------------
# 5. GET RECIPES (SMART SEARCH)
# -----------------------------
def get_recipes(predicted_label, calories, servings, recipe_type, protein_range, sugar_range):
    api_key = st.secrets["SPOONACULAR_API_KEY"]
    base_url = "https://api.spoonacular.com/recipes/complexSearch"
    label_variants = expand_label_variants(predicted_label)

    tried_terms = []

    # Step 1️⃣ — Progressive strict search
    for term in label_variants:
        tried_terms.append(term)
        params = {
            "apiKey": api_key,
            "query": term,
            "type": recipe_type.lower(),
            "minProtein": protein_range[0],
            "maxProtein": protein_range[1],
            "minSugar": sugar_range[0],
            "maxSugar": sugar_range[1],
            "maxCalories": calories,
            "number": 5
        }

        response = requests.get(base_url, params=params)
        data = response.json()
        results = data.get("results", [])

        if results:
            st.success(f"✅ Found recipes for **{term.capitalize()}**")
            return results

    # Step 2️⃣ — Relaxed search without filters
    st.warning(f"No exact match for {tried_terms}. Trying more general recipes...")
    for term in label_variants:
        relaxed_params = {"apiKey": api_key, "query": term, "number": 5}
        response = requests.get(base_url, params=relaxed_params)
        data = response.json()
        results = data.get("results", [])
        if results:
            st.info(f"🍴 Showing more general recipes for **{term}**")
            return results

    # Step 3️⃣ — Nothing found
    st.error(f"😔 Sorry, no recipes were found for **{predicted_label}**.")
    return []

# -----------------------------
# 6. STREAMLIT UI
# -----------------------------
uploaded_file = st.file_uploader("📸 Upload a fruit or vegetable image", type=["jpg", "png", "jpeg"])
calories = st.number_input("🔥 Max calories per serving", 50, 1000, 300)
servings = st.number_input("👥 Number of servings", 1, 10, 2)
recipe_type = st.selectbox("🍴 Recipe Type", ["Sweet", "Savory", "Drink", "Snack"])
protein_range = st.slider("💪 Protein range (grams)", 0, 50, (5, 20))
sugar_range = st.slider("🍬 Sugar range (grams)", 0, 50, (0, 15))

if st.button("Find Recipes"):
    if uploaded_file is not None:
        label, confidence = predict_image(uploaded_file)
        st.image(uploaded_file, caption="Your uploaded image", use_container_width=True)
        st.success(f"🧠 Predicted: **{label}** ({confidence*100:.2f}% confidence)")

        recipes = get_recipes(label, calories, servings, recipe_type, protein_range, sugar_range)

        if recipes:
            st.subheader("🍽 Recommended Recipes")
            for r in recipes:
                with st.container():
                    st.image(r.get("image", ""), width=250)
                    st.markdown(f"**{r['title']}**")
                    st.markdown(f"[🔗 View Recipe](https://spoonacular.com/recipes/{r['title'].replace(' ', '-')}-{r['id']})")
        else:
            st.info("Try adjusting your filters or upload another image.")
    else:
        st.warning("Please upload an image first.")
