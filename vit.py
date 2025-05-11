# models/vit.py
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import os
from utils.metrics import calculate_metrics, print_metrics_summary
from utils.visualization import save_visualizations

class ViT:
    def __init__(self, input_shape=(23, 4), num_classes=7, model_dir="saved_models/vit"):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.model = self._build_model()
        self.encoder = LabelEncoder()
        
    def _build_model(self):
        """Build the Vision Transformer model architecture"""
        inputs = layers.Input(shape=self.input_shape)
        
        # 1. Patch Creation
        patches = Patches(patch_size=4)(inputs)  # Split 23 timesteps into patches of 4
        
        # 2. Patch Encoding
        num_patches = (self.input_shape[0] // 4) * (self.input_shape[1] // 1)
        encoded_patches = PatchEncoder(num_patches, projection_dim=64)(patches)
        
        # 3. Transformer Blocks
        for _ in range(6):  # 6 transformer layers
            # Layer normalization and attention
            x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            attention_output = layers.MultiHeadAttention(
                num_heads=4, key_dim=64, dropout=0.1)(x1, x1)
            x2 = layers.Add()([attention_output, encoded_patches])
            
            # Layer normalization and MLP
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            x3 = self._mlp(x3, hidden_units=[128, 64], dropout_rate=0.1)
            encoded_patches = layers.Add()([x3, x2])
        
        # 4. Classification Head
        representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        representation = layers.Flatten()(representation)
        features = self._mlp(representation, hidden_units=[128, 64], dropout_rate=0.5)
        logits = layers.Dense(self.num_classes)(features)
        
        return Model(inputs=inputs, outputs=logits)
    
    def _mlp(self, x, hidden_units, dropout_rate):
        """MLP helper function"""
        for units in hidden_units:
            x = layers.Dense(units, activation=tf.nn.gelu)(x)
            x = layers.Dropout(dropout_rate)(x)
        return x

    def train(self, x_train, y_train, x_val, y_val, epochs=100, batch_size=64):
        """Training procedure with automatic model saving"""
        # Encode labels
        y_train_enc = self.encoder.fit_transform(y_train)
        y_val_enc = self.encoder.transform(y_val)
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(self.model_dir, "best_model.h5"),
                monitor="val_accuracy",
                save_best_only=True
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=10,
                restore_best_weights=True
            )
        ]
        
        # Compile
        self.model.compile(
            optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        # Train
        history = self.model.fit(
            x=x_train,
            y=y_train_enc,
            validation_data=(x_val, y_val_enc),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks
        )
        
        return history

    def evaluate(self, x_test, y_test):
        """Standardized evaluation"""
        y_test_enc = self.encoder.transform(y_test)
        y_pred = np.argmax(self.model.predict(x_test), axis=1)
        
        # Calculate metrics
        metrics = calculate_metrics(y_test_enc, y_pred, "ViT")
        print_metrics_summary(metrics)
        
        # Generate visualizations
        save_visualizations(
            model=self.model,
            x_data=x_test,
            y_true=y_test_enc,
            y_pred=y_pred,
            model_name="ViT"
        )
        
        return metrics

    def save(self, model_name="vit_model"):
        """Save the complete model"""
        save_path = os.path.join(self.model_dir, f"{model_name}.h5")
        self.model.save(save_path)
        print(f"Model saved to {save_path}")

    @classmethod
    def load(cls, model_path):
        """Load a saved model"""
        model = tf.keras.models.load_model(model_path)
        vit = cls(input_shape=model.input_shape[1:], 
                 num_classes=model.output_shape[-1])
        vit.model = model
        return vit


# Helper Layers
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, x):
        batch_size = tf.shape(x)[0]
        # Reshape into patches
        patches = tf.reshape(
            x,
            [batch_size, -1, self.patch_size * x.shape[-1]]
        )
        return patches

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
