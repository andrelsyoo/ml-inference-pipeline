# export_model.py
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

print("TensorFlow version:", tf.__version__)
print("Downloading MobileNetV2 from TensorFlow Hub...")

# Load MobileNetV2 from TF Hub
model_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/5"

# Create a simple model
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
    hub.KerasLayer(model_url, trainable=False)
])

# Build the model
model.build([None, 224, 224, 3])

print("Model loaded successfully!")
print(model.summary())

# Save in TensorFlow Serving format
save_path = "models/mobilenet/1"
print(f"\nSaving model to {save_path}...")
tf.saved_model.save(model, save_path)

print("✅ Model exported successfully!")
print(f"Model saved at: {save_path}")