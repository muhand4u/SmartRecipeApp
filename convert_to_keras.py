import tensorflow as tf

model = tf.keras.models.load_model("models/fruit_model.h5")
model.save("models/best_model.keras")

print("✅ Model successfully converted to models/best_model.keras")
