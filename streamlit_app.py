import streamlit as st

st.title("🍲 Smart Recipe Recommender - Version 1.0")
st.write("Welcome, Mohanad! 👋")
st.write("This is your first Streamlit app deployed from GitHub.")

# User inputs
image = st.file_uploader("📸 Upload a food image", type=["jpg", "png"])
calories = st.number_input("🔥 Calories per serving", 50, 1000, 300)
servings = st.number_input("👥 Number of servings", 1, 10, 2)
recipe_type = st.selectbox("🍴 Recipe Type", ["Sweet", "Savory", "Drink", "Snack"])
protein_range = st.slider("💪 Protein (grams)", 0, 50, (5, 20))
sugar_range = st.slider("🍬 Sugar (grams)", 0, 50, (0, 15))

if st.button("Find Recipes"):
    st.success(f"You searched for a {recipe_type} recipe with about {calories} kcal/serving!")
    st.info(f"Protein range: {protein_range} g | Sugar range: {sugar_range} g")
    st.image("https://upload.wikimedia.org/wikipedia/commons/2/2c/Banana_Smoothie.jpg", caption="Sample Result 🍌")