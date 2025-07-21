# if it don't work on vscode try on google coolabe, maybe it's a little problem with dataset dowloads

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models
import numpy as np

# Loads the dataset with labels
dataset, info = tfds.load("cats_vs_dogs", with_info=True, as_supervised=True)
full_dataset = dataset["train"]

# Preprocessing
def preprocess(image, label):
    # Processes the image so it can be used by the model
    image = tf.image.resize(image, (224, 224))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# Applies the preprocessing function to each image
full_dataset = full_dataset.map(preprocess)

# Split: 70% training, 15% validation, 15% testing
total_size = info.splits["train"].num_examples
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)

# Dataset splitting
full_dataset = full_dataset.shuffle(1000, seed=42)
train_dataset = full_dataset.take(train_size).batch(32)
val_dataset = full_dataset.skip(train_size).take(val_size).batch(32)
test_dataset = full_dataset.skip(train_size + val_size).batch(32)

# Loads the MobileNetV2 base without the final layer, pre-trained on ImageNet
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    # Removes the final layer (the head)
    include_top=False,
    weights='imagenet'
)

# Freezes the base (prevents its weights from being updated during initial training)
base_model.trainable = False

# Builds the full model with the base + new head
model = tf.keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')  # Binary output
])

# Compiles the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# Train only the new head for 5 epochs
history = model.fit(train_dataset,
                    validation_data=val_dataset,
                    epochs=5)

# Saves the model so it can be reused without retraining
model.save("modelo_gatos_cachorros.h5")
