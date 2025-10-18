import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model 
import numpy as np
import json
import os
from PIL import Image
import requests

# -----------------------------------------
# 1. LOAD MODEL + LABEL MAP (AUTO DETECT FORMAT)
# -----------------------------------------

@st.cache_resource
def load_model_and_labels():
    model_path = "models/best_model_tf_compatible.h5"
    labels_path = "models/class_indices.json"

    # ✅ Safe loading with clear error handling
    if not os.path.exists(model_path):
        st.error("❌ Model file not found. Please upload or push models/best_model_tf_compatible.h5.")
        st.stop()

    try:
        # Try standard TF load first
        model = tf.keras.models.load_model(model_path, compile=False)
        st.info("✅ Loaded TensorFlow-compatible model (.h5)")
    except Exception as e:
        # Try fallback legacy format if first fails
        st.warning(f"⚠️ Standard load failed: {e}\nTrying legacy loader...")
        from keras.models import load_model as legacy_load
        model = legacy_load(model_path, compile=False)
        st.info("✅ Loaded model via legacy Keras loader")

    if not os.path.exists(labels_path):
        st.error("❌ Missing class_indices.json file.")
        st.stop()

    with open(labels_path, "r") as f:
        class_indices = json.load(f)

    # Invert the label map (class_idx → label_name)
    index_to_label = {v: k for k, v in class_indices.items()}

    return model, index_to_label

# -----------------------------------------
# 2. IMAGE PREPROCESSING FUNCTION
# -----------------------------------------
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# -----------------------------------------
# 3. CLEAN LABEL (MAKE IT GENERAL)
# -----------------------------------------
def clean_label(label):
    # Remove underscores, numbers, and extra words
    words = ''.join([c if c.isalpha() or c == ' ' else ' ' for c in label])
    tokens = [w for w in words.split() if len(w) > 2]
    return ' '.join(tokens[:3])  # limit to first 3 words (e.g., "banana lady finger")


# -----------------------------------------
# 4. SPOONACULAR RECIPE FETCH
# -----------------------------------------
def get_recipes(ingredient, calories, servings, recipe_type, protein_range, sugar_range):
    try:
        api_key = st.secrets["SPOONACULAR_API_KEY"]
    except Exception:
        st.warning("⚠️ Spoonacular API key not found in Streamlit secrets.")
        return []

    url = (
        f"https://api.spoonacular.com/recipes/complexSearch?"
        f"query={ingredient}&type={recipe_type}&number=5"
        f"&minCalories={max(0, calories - 100)}&maxCalories={calories + 100}"
        f"&apiKey={api_key}"
    )
    response = requests.get(url)
    data = response.json()

    if "results" not in data or not data["results"]:
        st.warning(f"No direct recipes found for '{ingredient}'. Trying broader search...")
        # broader search using first keyword only
        first_word = ingredient.split()[0]
        url = f"https://api.spoonacular.com/recipes/complexSearch?query={first_word}&number=5&apiKey={api_key}"
        response = requests.get(url)
        data = response.json()

    return data.get("results", [])


# -----------------------------------------
# 5. STREAMLIT UI
# -----------------------------------------
st.title("🍲 Smart Recipe Recommender - AI Powered")
st.write("Upload a fruit or vegetable image and get recipes tailored to your filters!")

image = st.file_uploader("📸 Upload an image", type=["jpg", "png", "jpeg"])
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
    st.image(img, caption="Your uploaded image", use_container_width=True)

    img_array = preprocess_image(img)
    preds = model.predict(img_array)
    pred_idx = np.argmax(preds)
    pred_label_raw = index_to_label[pred_idx]
    confidence = np.max(preds) * 100

    cleaned_label = clean_label(pred_label_raw)
    st.success(f"🎯 Predicted: {cleaned_label} ({confidence:.2f}% confidence)")

    recipes = get_recipes(cleaned_label, calories, servings, recipe_type, protein_range, sugar_range)

    if recipes:
        st.subheader("🍽️ Suggested Recipes:")
        for r in recipes:
            st.markdown(f"### [{r['title']}]({r.get('sourceUrl', '#')})")
            if r.get("image"):
                st.image(r["image"], width=250)
    else:
        st.error(f"😔 Sorry, no recipes found for '{cleaned_label}'. Try a different filter or item.")
