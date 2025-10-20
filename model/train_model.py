# model/train_model.py
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# =====================================================
# 1. PATH SETUP (portable)
# =====================================================
base_dir = os.path.dirname(os.path.abspath(__file__))
data_root = os.path.join(base_dir, "../Data")
train_dir = os.path.join(data_root, "Training")
val_dir = os.path.join(data_root, "Test")

models_dir = os.path.join(base_dir, "checkpoints")
os.makedirs(models_dir, exist_ok=True)

print(f"✅ Training folder: {train_dir}")
print(f"✅ Validation folder: {val_dir}")

# =====================================================
# 2. DEVICE CHECK
# =====================================================
print("\n🔍 Checking device availability:")
print("GPU found ✅" if tf.config.list_physical_devices("GPU") else "⚠️ Running on CPU")

# =====================================================
# 3. IMAGE DATA GENERATORS
# =====================================================
IMG_SIZE = 224
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)
val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# =====================================================
# 4. LOAD OR CREATE MODEL
# =====================================================
model_path = os.path.join(models_dir, "best_model.h5")

if os.path.exists(model_path):
    print("🔁 Loading previously saved model...")
    model = load_model(model_path)
    base_model = None
else:
    print("🚀 Creating new MobileNetV2 model...")
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(train_gen.num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

model.summary()

# =====================================================
# 5. CALLBACKS
# =====================================================
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-5, verbose=1),
    ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, verbose=1)
]

# =====================================================
# 6. TRAINING
# =====================================================
EPOCHS = 10
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks
)

# =====================================================
# 7. FINE-TUNE LAST 20 LAYERS
# =====================================================
if base_model is not None:
    print("\n🔧 Unfreezing last 20 layers for fine-tuning...")
    for layer in base_model.layers[-20:]:
        layer.trainable = True

    model.compile(optimizer=Adam(learning_rate=1e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history_fine = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10,
        callbacks=callbacks
    )

# =====================================================
# 8. SAVE FINAL MODEL + CLASS MAP
# =====================================================
final_model_path = os.path.join(base_dir, "fruit_model.h5")
labels_path = os.path.join(base_dir, "labels.json")

model.save(final_model_path)
with open(labels_path, "w") as f:
    json.dump(train_gen.class_indices, f)

print(f"\n✅ Model saved: {final_model_path}")
print(f"✅ Labels saved: {labels_path}")

# =====================================================
# 9. EVALUATION
# =====================================================
val_loss, val_acc = model.evaluate(val_gen)
print(f"\nValidation Accuracy: {val_acc:.4f}")

y_pred = np.argmax(model.predict(val_gen), axis=1)
y_true = val_gen.classes
print("\nClassification Report:\n",
      classification_report(y_true, y_pred, target_names=list(val_gen.class_indices.keys())))

# =====================================================
# 10. PLOTS
# =====================================================
plt.figure(figsize=(8, 4))
plt.plot(history.history['accuracy'], label='Train (Frozen)')
plt.plot(history.history['val_accuracy'], label='Val (Frozen)')
plt.legend()
plt.title('Model Accuracy Over Epochs')
plt.show()
