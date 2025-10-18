import tensorflow as tf

model = tf.keras.models.load_model("models/best_model_final.h5")
model.save("models/best_model_streamlit.keras")