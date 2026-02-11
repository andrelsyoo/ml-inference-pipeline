# export_model.py
import tensorflow as tf
import tensorflow_hub as hub
import os

print("=" * 60)
print("TensorFlow Model Export Script")
print("=" * 60)
print(f"TensorFlow version: {tf.__version__}")
print(f"TensorFlow Hub version: {hub.__version__}")
print()

# Create model directory if it doesn't exist
save_path = "models/mobilenet/1"
os.makedirs(save_path, exist_ok=True)

print("Downloading MobileNetV2 from TensorFlow Hub...")
print("This may take a few minutes on first run...")
print()

# Load model directly from TF Hub and save it
model_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/5"

print(f"Loading model from: {model_url}")
# Download the model
module = hub.load(model_url)

print("Model loaded successfully!")
print()

# Save in TensorFlow Serving format
print(f"Saving model to {save_path}...")
tf.saved_model.save(module, save_path)

print()
print("✅ Model exported successfully!")
print(f"📁 Model location: {os.path.abspath(save_path)}")
print()
print("Model details:")
print(f"  - Input: 224x224 RGB images")
print(f"  - Output: 1001 class probabilities (ImageNet classes)")
print(f"  - Format: TensorFlow SavedModel (ready for TF Serving)")
print()
print("Next steps:")
print("  1. Deactivate venv: deactivate")
print("  2. Download test image: curl -o test/cat.jpg https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/480px-Cat03.jpg")
print("  3. Run: docker-compose up --build")
print("  4. Test: curl -X POST -F 'file=@test/cat.jpg' http://localhost:8000/predict")
print("=" * 60)