import streamlit as st
import tensorflow as tf
import numpy as np
import json
import os
from PIL import Image
import requests

# =====================================================
# 1. LOAD MODEL + LABEL MAP
# =====================================================
@st.cache_resource
def load_model_and_labels():
    model_path = os.path.join("model", "fruit_model.h5")
    labels_path = os.path.join("model", "labels.json")

    if not os.path.exists(model_path):
        st.error("❌ Model file not found. Please train and save fruit_model.h5 in /model.")
        st.stop()
    if not os.path.exists(labels_path):
        st.error("❌ labels.json not found in /model.")
        st.stop()

    model = tf.keras.models.load_model(model_path)
    with open(labels_path, "r") as f:
        class_indices = json.load(f)
    index_to_label = {v: k for k, v in class_indices.items()}
    st.info("✅ Model and labels loaded successfully!")
    return model, index_to_label


# =====================================================
# 2. IMAGE PREPROCESSING
# =====================================================
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# =====================================================
# 3. CLEAN LABEL (MAKE IT GENERAL)
# =====================================================
def clean_label(label):
    words = ''.join([c if c.isalpha() or c == ' ' else ' ' for c in label])
    tokens = [w for w in words.split() if len(w) > 2]
    return ' '.join(tokens[:3])


# =====================================================
# 4. SPOONACULAR RECIPE FETCH
# =====================================================
def get_recipes(ingredient, calories, servings, recipe_type, protein_range, sugar_range):
    try:
        api_key = st.secrets["SPOONACULAR_API_KEY"]
    except Exception:
        st.warning("⚠️ Spoonacular API key not found in Streamlit secrets.")
        return []

    base_url = "https://api.spoonacular.com/recipes/complexSearch"
    params = {
        "query": ingredient,
        "type": recipe_type.lower() if recipe_type != "Any" else "",
        "number": 5,
        "minCalories": max(0, calories - 100),
        "maxCalories": calories + 100,
        "apiKey": api_key
    }

    response = requests.get(base_url, params=params)
    data = response.json()

    if "results" not in data or not data["results"]:
        st.warning(f"No recipes found for '{ingredient}'. Trying broader search...")
        first_word = ingredient.split()[0]
        response = requests.get(base_url, params={"query": first_word, "number": 5, "apiKey": api_key})
        data = response.json()

    return data.get("results", [])


# =====================================================
# 5. STREAMLIT UI
# =====================================================
st.title("🥗 Smart Recipe Recommender — AI-Powered")
st.write("Upload a fruit or vegetable image and get recipes tailored to your preferences!")

image = st.file_uploader("📸 Upload an image", type=["jpg", "jpeg", "png"])
calories = st.number_input("🔥 Calories per serving", 50, 1000, 300)
servings = st.number_input("👥 Number of servings", 1, 10, 2)
recipe_type = st.selectbox("🍴 Recipe Type", ["Sweet", "Savory", "Snack", "Drink", "Any"])
protein_range = st.slider("💪 Protein (grams)", 0, 50, (5, 20))
sugar_range = st.slider("🍬 Sugar (grams)", 0, 50, (0, 15))

model, index_to_label = load_model_and_labels()

if st.button("🔍 Find Recipes"):
    if not image:
        st.warning("Please upload an image first.")
        st.stop()

    img = Image.open(image).convert("RGB")
    st.image(img, caption="Uploaded image", use_container_width=True)

    img_array = preprocess_image(img)
    preds = model.predict(img_array)
    pred_idx = np.argmax(preds)
    pred_label_raw = index_to_label[pred_idx]
    confidence = float(np.max(preds)) * 100

    cleaned_label = clean_label(pred_label_raw)
    st.success(f"🎯 Predicted: {cleaned_label} ({confidence:.2f}% confidence)")

    recipes = get_recipes(cleaned_label, calories, servings, recipe_type, protein_range, sugar_range)

    if recipes:
        st.subheader("🍽️ Suggested Recipes:")
        for r in recipes:
            title = r.get("title", "Untitled Recipe")
            source_url = r.get("sourceUrl", "#")
            st.markdown(f"### [{title}]({source_url})")
            if r.get("image"):
                st.image(r["image"], width=250)
    else:
        st.error(f"😔 No recipes found for '{cleaned_label}'. Try adjusting filters.")
