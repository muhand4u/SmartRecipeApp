# 🥗 SmartRecipeApp — AI-Powered Food Recommender

### Graduate Data Science Project — DTSC691  
Author: **Mohanad Mallooki**

---

## 🧠 Overview

**SmartRecipeApp** is a web-based application that combines **deep learning image classification** and **recipe recommendation**.  
It identifies fruits or vegetables from an uploaded image and recommends recipes containing that ingredient using the **Spoonacular API**.

This project integrates:
- 🧩 **TensorFlow (MobileNetV2)** — Transfer learning model for image recognition  
- 🧪 **Streamlit** — Interactive web application for deployment  
- 🌐 **Spoonacular API** — Recipe search and filtering by calories, protein, sugar, and type  
- 📊 **Python data pipeline** — Image preprocessing, feature extraction, and prediction logic  

---

## 🚀 Project Structure



SmartRecipeApp/
│
├── model/
│ ├── train_model.py # Training script using MobileNetV2 transfer learning
│ ├── fruit_model.h5 # Trained model file (generated after training)
│ └── labels.json # Class label map (generated after training)
│
├── Data/
│ ├── Training/ # Training images (each class = folder name)
│ └── Test/ # Validation images
│
├── streamlit_app.py # Main Streamlit web app
├── requirements.txt # Python dependencies (TensorFlow + Streamlit)
├── README.md # Project documentation
└── .streamlit/
└── secrets.toml # API key storage (not committed to Git)


---

## ⚙️ How It Works

1. **Model Training**
   - Uses **MobileNetV2** pretrained on ImageNet.
   - Fine-tunes the last layers to classify fruits and vegetables.
   - The model is trained using images organized by folder structure:
     ```
     Data/Training/apple/
     Data/Training/banana/
     Data/Training/tomato/
     ```
   - The trained model is saved as:
     ```
     model/fruit_model.h5
     model/labels.json
     ```

2. **Prediction Phase**
   - The Streamlit app loads the trained model and class labels.
   - Users upload an image of a fruit or vegetable.
   - The model predicts the ingredient name (e.g., “Banana”).
   - The app queries **Spoonacular API** to recommend recipes using that ingredient.

3. **Recipe Filtering**
   - Users can filter recipes by:
     - Calories per serving  
     - Number of servings  
     - Protein range  
     - Sugar range  
     - Recipe type (Sweet, Savory, Snack, Drink, etc.)

4. **Results Display**
   - Streamlit displays a list of matching recipes with titles, links, and images.

---

## 🧰 Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/SmartRecipeApp.git
cd SmartRecipeApp

2️⃣ Create Virtual Environment
python -m venv .venv
.venv\Scripts\activate     # (Windows)
source .venv/bin/activate  # (Mac/Linux)

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Prepare Dataset

Organize your dataset into subfolders (each representing a class):

Data/
├── Training/
│   ├── apple/
│   ├── banana/
│   └── carrot/
└── Test/
    ├── apple/
    ├── banana/
    └── carrot/

5️⃣ Train the Model
cd model
python train_model.py


This generates fruit_model.h5 and labels.json.

6️⃣ Run the Streamlit App
cd ..
streamlit run streamlit_app.py

☁️ Deployment (Streamlit Cloud)

Push your project to GitHub.

Go to Streamlit Cloud
.

Connect your GitHub repo and select the streamlit_app.py file.

Under App Settings → Advanced, set:

Python version: 3.11

Dependencies: Use requirements.txt

Add your Spoonacular API key in Streamlit’s Secrets Manager:

[general]
SPOONACULAR_API_KEY = "your_actual_api_key"

🧠 Model Architecture (MobileNetV2)
Input (224×224×3)
↓
MobileNetV2 (pretrained on ImageNet)
↓
GlobalAveragePooling2D
↓
Dense(256, ReLU)
↓
Dropout(0.5)
↓
Dense(N, Softmax)


Base model: MobileNetV2 (frozen for first training phase)

Optimizer: Adam (lr = 0.001 → fine-tuned at 1e-5)

Loss: Categorical Cross-Entropy

Epochs: 10 (frozen) + 10 (fine-tuned)

Metrics: Accuracy

📊 Example Output

Input: Image of a banana 🍌
Model Prediction: Banana (98.2% confidence)
Recipe Results:

Banana Smoothie

Chocolate Banana Bread

Peanut Butter Banana Oats

🧾 Requirements
Package	Version
TensorFlow	2.15.0
Streamlit	1.39.0
NumPy	Latest
Pillow	Latest
Requests	Latest
🧩 Future Enhancements

🍳 Add ingredient detection for mixed dishes.

🧠 Incorporate nutritional prediction from image (calories, macros).

🗣️ Enable voice or text input to combine ingredients.

📱 Optimize for mobile view.

📜 License

MIT License — Free for educational and research use.

🙌 Acknowledgements

TensorFlow Team for MobileNetV2 pretrained weights

Spoonacular API for recipe data

Streamlit for rapid web app development

Eastern University — DTSC691 for providing the academic foundation for this capstone project


---

## 🪄 Notes for You

- You can edit the author line or add your student ID for submission.
- The “Future Enhancements” section is optional but adds a strong academic touch.
- Once this file is saved as `README.md` in your repo root, GitHub and Streamlit Cloud will both display it beautifully.

---

Would you like me to add a short **“Project Abstract”** section at the top (for academic submission — about 5–6 lines summarizing purpose, methods, and outcome)? It helps for DTSC691 documentation.