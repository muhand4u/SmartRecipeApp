import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import requests

# ----------------------------
# 1. PAGE SETUP
# ----------------------------
st.title("🍲 Smart Recipe Recommender - Version 3.0")
st.write("Now featuring real-time recipe suggestions based on your detected food and filters!")

# ----------------------------
# 2. LOAD MODEL + LABELS
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
# 4. SPOONACULAR API SETTINGS
# ----------------------------
API_KEY = st.secrets["api_keys"]["spoonacular"]  # <-- replace this with your real key
API_URL = "https://api.spoonacular.com/recipes/complexSearch"

def get_recipes(ingredient, recipe_type, calories, protein_min, protein_max, sugar_min, sugar_max, servings):
    params = {
        "apiKey": API_KEY,
        "query": ingredient,
        "type": recipe_type.lower(),
        "maxCalories": calories,
        "number": 5,  # how many recipes to show
        "addRecipeInformation": True,
    }
    response = requests.get(API_URL, params=params)
    if response.status_code == 200:
        data = response.json()
        return data.get("results", [])
    else:
        st.error(f"API Error {response.status_code}: {response.text}")
        return []

# ----------------------------
# 5. DETECT & RECOMMEND
# ----------------------------
if st.button("🔍 Identify & Find Recipes"):
    if image is not None:
        # Step 1: Predict ingredient
        img = Image.open(image).convert("RGB").resize((100,100))
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
        preds = model.predict(img_array)
        class_id = int(np.argmax(preds))
        confidence = float(np.max(preds) * 100)
        ingredient = labels[class_id]

        st.image(image, caption=f"📷 Uploaded Image — Detected: {ingredient}", use_container_width=True)
        st.success(f"✅ Predicted: **{ingredient}** ({confidence:.2f}% confidence)")

        # Step 2: Get recipes from API
        with st.spinner("Fetching recipes... 🍳"):
            recipes = get_recipes(
                ingredient,
                recipe_type,
                calories,
                protein_range[0],
                protein_range[1],
                sugar_range[0],
                sugar_range[1],
                servings
            )

        # Step 3: Display results
        if recipes:
            st.subheader(f"🍽️ Recipes for {ingredient.capitalize()} ({len(recipes)} found)")
            for recipe in recipes:
                st.markdown(f"### [{recipe['title']}]({recipe['sourceUrl']})")
                if "image" in recipe:
                    st.image(recipe["image"], use_container_width=True)
                if "readyInMinutes" in recipe:
                    st.caption(f"⏱️ Ready in {recipe['readyInMinutes']} minutes")
                if "summary" in recipe:
                    summary = recipe["summary"].replace("<b>", "").replace("</b>", "")
                    st.write(summary[:200] + "...")
                st.markdown("---")
        else:
            st.warning("No recipes found — try adjusting filters or using a common fruit/vegetable.")
    else:
        st.warning("⚠️ Please upload an image first.")
