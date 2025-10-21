import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import json
import seaborn as sns

# =====================================================
# 1. PATH SETUP
# =====================================================
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_root = os.path.join(base_dir, "Data")
train_dir = os.path.join(data_root, "Training")
val_dir = os.path.join(data_root, "Test")

model_dir = "model"
os.makedirs(model_dir, exist_ok=True)

print(f"✅ Training folder: {train_dir}")
print(f"✅ Validation folder: {val_dir}")

# =====================================================
# 2. GPU / DEVICE CHECK
# =====================================================
print("\n🔍 Checking device availability:")
if tf.config.list_physical_devices('GPU'):
    print("✅ GPU detected")
else:
    print("⚠️ No GPU detected — running on CPU")

# =====================================================
# 3. IMAGE DATA GENERATORS (STRONGER AUGMENTATION)
# =====================================================
IMG_SIZE = 224
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    brightness_range=[0.7, 1.3],
    horizontal_flip=True,
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator(rescale=1.0/255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# =====================================================
# 4. COMPUTE CLASS WEIGHTS (BALANCE CLASSES)
# =====================================================
classes = np.unique(train_gen.classes)
weights = compute_class_weight(class_weight="balanced", classes=classes, y=train_gen.classes)
class_weights = dict(zip(classes, weights))
print("\n📊 Class Weights:", class_weights)

# =====================================================
# 5. LOAD OR CREATE MODEL (MobileNetV2)
# =====================================================
model_path = os.path.join(model_dir, "fruit_model.h5")

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
    base_model.trainable = False  # freeze for initial training

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    predictions = Dense(train_gen.num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

model.summary()

# =====================================================
# 6. CALLBACKS
# =====================================================
callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, min_lr=1e-5, verbose=1),
    ModelCheckpoint(model_path, monitor="val_accuracy", save_best_only=True, verbose=1)
]

# =====================================================
# 7. TRAINING (FROZEN BASE)
# =====================================================
EPOCHS = 15
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)

# =====================================================
# 8. FINE-TUNE (UNFREEZE LAST 50 LAYERS)
# =====================================================
if base_model is not None:
    print("\n🔧 Fine-tuning last 50 layers...")
    for layer in base_model.layers[-50:]:
        layer.trainable = True

    model.compile(optimizer=Adam(learning_rate=1e-5),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    EPOCHS_FINE = 10
    history_fine = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_FINE,
        class_weight=class_weights,
        callbacks=callbacks
    )
else:
    print("✅ Model already fine-tuned.")

# =====================================================
# 9. SAVE MODEL + LABEL MAP
# =====================================================
model.save(model_path)
with open(os.path.join(model_dir, "labels.json"), "w") as f:
    json.dump(train_gen.class_indices, f)

print("\n✅ Model and labels saved to /model")

# =====================================================
# 🔍 10. EVALUATION + CONFUSION MATRIX
# =====================================================
val_loss, val_acc = model.evaluate(val_gen)
print(f"\n📈 Validation Accuracy: {val_acc:.4f}")

y_pred = np.argmax(model.predict(val_gen), axis=1)
y_true = val_gen.classes

# Classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=list(val_gen.class_indices.keys())))

# Confusion Matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=val_gen.class_indices.keys(),
            yticklabels=val_gen.class_indices.keys())
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()
