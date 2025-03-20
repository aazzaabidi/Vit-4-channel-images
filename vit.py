import os
from keras import layers
from keras import ops
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import tensorflow as tf
from tensorflow import keras

train_data_path = ""
train_labels_path = ""
valid_data_path = ""
valid_labels_path = ""
test_data_path = ""
test_labels_path = ""

# Load the NumPy files
x_train = np.load(train_data_path)
y_train = np.load(train_labels_path)
x_valid = np.load(valid_data_path)
y_valid = np.load(valid_labels_path)
x_test = np.load(test_data_path)
y_test = np.load(test_labels_path)

# Print dataset shapes
print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
print(f"x_valid shape: {x_valid.shape} - y_valid shape: {y_valid.shape}")
print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")

num_classes = len(np.unique(train_y))
input_shape = train_x.shape[1:]
patch_size = 4  
num_patches = (input_shape[0] // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_layers = 6
transformer_units = [projection_dim * 2, projection_dim]
mlp_head_units = [128, 64]  # Dense layers pour la classification

print(f"train_x shape: {train_x.shape} - train_y shape: {train_y.shape}")
print(f"Nombre de classes: {num_classes}")


unique_values = np.unique(train_y)
print(unique_values)

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        return tf.reshape(patches, [batch_size, -1, self.patch_size * self.patch_size * input_shape[-1]])

#MLP
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=keras.activations.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

#batch creation as layer
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        input_shape = ops.shape(images)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        patches = keras.ops.image.extract_patches(images, size=self.patch_size)
        patches = ops.reshape(
            patches,
            (
                batch_size,
                num_patches_h * num_patches_w,
                self.patch_size * self.patch_size * channels,
            ),
        )
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config
# patch encoding layer


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = ops.expand_dims(
            ops.arange(start=0, stop=self.num_patches, step=1), axis=0
        )
        projected_patches = self.projection(patch)
        encoded = projected_patches + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches})
        return config

  #model building
def create_vit_classifier():
    inputs = keras.Input(shape=input_shape)
    # Create patches.
    patches = Patches(patch_size)(inputs)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
    logits = layers.Dense(num_classes)(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model
def run_experiment(model):
    optimizer = keras.optimizers.AdamW(
        learning_rate=1e-4, weight_decay=1e-3
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    checkpoint_filepath = "/tmp/checkpoint.weights.h5"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    history = model.fit(
        x=train_x,
        y=train_y,
        batch_size=16,
        epochs=300,
        validation_data=(valid_x, valid_y), 
        callbacks=[checkpoint_callback],
    )

    model.load_weights(checkpoint_filepath)

    y_pred = np.argmax(model.predict(test_x), axis=1)

    # Compute Accuracy
    accuracy = accuracy_score(test_y, y_pred)
    print(f"Test Accuracy: {round(accuracy * 100, 2)}%")

    # Compute F1-score
    f1 = f1_score(test_y, y_pred, average="macro")
    print(f"Test F1-score (macro): {round(f1, 4)}")

    # Compute Confusion Matrix
    conf_matrix = confusion_matrix(test_y, y_pred)

    plt.figure(figsize=(10, 6))
    sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()

    return history

# Make sure create_vit_classifier() is defined
vit_classifier = create_vit_classifier()
history = run_experiment(vit_classifier)
