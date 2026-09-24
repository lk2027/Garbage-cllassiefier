import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import json
import os

# ===============================
# Paths
# ===============================
train_dir = "dataset/train"
val_dir = "dataset/val"

# ===============================
# Parameters
# ===============================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30  # full fine-tuning epochs

# ===============================
# Data augmentation
# ===============================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator(rescale=1./255)

# ===============================
# Generators
# ===============================
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# ===============================
# Load pre-trained MobileNetV2
# ===============================
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False  # freeze base model initially

# ===============================
# Build model
# ===============================
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(train_generator.num_classes, activation="softmax")
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# ===============================
# Callbacks
# ===============================
callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_accuracy", factor=0.3, patience=3, verbose=1),
    ModelCheckpoint("waste_model_best.keras", save_best_only=True, monitor="val_accuracy", mode="max")
]

# ===============================
# Train only the head first
# ===============================
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    callbacks=callbacks
)

# ===============================
# Fine-tune deeper layers
# ===============================
base_model.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

history_finetune = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ===============================
# Save final model
# ===============================
MODEL_PATH = "GarbageClassifier7.h5"
model.save(MODEL_PATH)
print(f"✅ Model training complete and saved as {MODEL_PATH}")

# ===============================
# Save class indices
# ===============================
CLASS_INDICES_PATH = "class_indices.json"
class_indices = train_generator.class_indices  # {class_name: index}
with open(CLASS_INDICES_PATH, "w") as f:
    json.dump(class_indices, f)
print(f"✅ Class indices saved as {CLASS_INDICES_PATH}")
