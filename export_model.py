# export_model.py
import tensorflow as tf
import tensorflow_hub as hub
import os
import shutil

print("=" * 60)
print("TensorFlow Model Export Script (Fixed)")
print("=" * 60)
print(f"TensorFlow version: {tf.__version__}")
print(f"TensorFlow Hub version: {hub.__version__}")
print()

# Remove old model if exists
save_path = "models/mobilenet/1"
if os.path.exists(save_path):
    print(f"Removing old model at {save_path}...")
    shutil.rmtree(save_path)

os.makedirs(save_path, exist_ok=True)

print("Downloading MobileNetV2 from TensorFlow Hub...")
print("This may take a few minutes...")
print()

model_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/5"

# Load the hub module
print(f"Loading model from: {model_url}")
hub_module = hub.load(model_url)

print("Model loaded successfully!")
print()

# Create a concrete function with proper signature
@tf.function(input_signature=[tf.TensorSpec(shape=[None, 224, 224, 3], dtype=tf.float32, name='inputs')])
def serving_fn(inputs):
    return hub_module(inputs)

# Save with proper serving signature
print(f"Saving model with serving signature to {save_path}...")
tf.saved_model.save(
    hub_module,
    save_path,
    signatures={
        'serving_default': serving_fn
    }
)

print()
print("✅ Model exported successfully with serving signature!")
print(f"📁 Model location: {os.path.abspath(save_path)}")
print()
print("Model details:")
print(f"  - Input: 224x224 RGB images (float32, normalized [0,1])")
print(f"  - Output: 1001 class probabilities (ImageNet classes)")
print(f"  - Format: TensorFlow SavedModel with 'serving_default' signature")
print(f"  - Signature name: serving_default")
print()
print("Next steps:")
print("  1. Restart docker: docker-compose down && docker-compose up")
print("  2. Test: curl -X POST -F 'file=@test/cat.jpg' http://localhost:8000/predict")
print("=" * 60)