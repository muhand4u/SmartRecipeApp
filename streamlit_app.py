import streamlit as st
import tensorflow as tf
import numpy as np
import json
import os
from PIL import Image, UnidentifiedImageError
import requests
import itertools
import re

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
IMG_SIZE = 224

def preprocess_image(img):
    img = img.resize((IMG_SIZE, IMG_SIZE))
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
    """
    Fetches recipes from the Spoonacular API based on predicted ingredient and filters.
    """
    try:
        api_key = st.secrets["SPOONACULAR_API_KEY"]
    except Exception:
        st.warning("⚠️ Spoonacular API key not found in Streamlit secrets.")
        return []

    # 🔹 Step 1: Generate all search term combinations from ingredient
    tokens = [t.strip().lower() for t in ingredient.split() if len(t.strip()) > 2]
    if not tokens:
        return []

    search_terms = set()
    for r in range(1, len(tokens) + 1):
        for combo in itertools.permutations(tokens, r):
            term = " ".join(combo)
            search_terms.add(term)

    st.info("🍽️ Let’s find some delicious ideas that match your photo...")

    best_result = []
    best_term = None
    most_hits = 0

    # 🔹 Step 2: Try each combination until we find recipes
    for term in search_terms:
        # Build query using all filters (calories, servings, protein, sugar)
        url = (
            f"https://api.spoonacular.com/recipes/complexSearch?"
            f"query={term}&type={recipe_type}&number=10"
            f"&minCalories={max(0, calories - 100)}&maxCalories={calories + 100}"
            f"&minProtein={protein_range[0]}&maxProtein={protein_range[1]}"
            f"&minSugar={sugar_range[0]}&maxSugar={sugar_range[1]}"
            f"&apiKey={api_key}"
        )

        try:
            response = requests.get(url)
            data = response.json()
        except Exception:
            continue

        hits = len(data.get("results", []))
        if hits > most_hits:
            most_hits = hits
            best_result = data["results"]
            best_term = term

    # 🔹 Step 3: Display which term matched best
    if most_hits > 0:
        st.success(f"🎯 Found best fit {most_hits} recipes.")
        return best_result
    else:
        return []


# =====================================================
# 5. STREAMLIT UI
# =====================================================
st.title("🥗 Smart Recipe Recommender — AI-Powered - By Mohanad Mallooki")
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

    try:
        image.seek(0)  # Reset file pointer
        img = Image.open(image)
        img = img.convert("RGB")

        # 🖼️ Display uploaded image (with fallback for older Streamlit versions)
        try:
            st.image(img, caption="Uploaded image", use_container_width=True)
        except TypeError:
            st.image(img, caption="Uploaded image")

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

    except UnidentifiedImageError:
        st.error("❌ The uploaded file is not a valid image. Please upload a JPG or PNG.")
    except Exception as e:
        st.error(f"Unexpected error: {e}")