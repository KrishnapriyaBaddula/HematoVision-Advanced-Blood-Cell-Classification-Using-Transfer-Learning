# test_model.py
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

# Load model
model = tf.keras.models.load_model("models/blood_cell_classifier.h5")

# Load class names
with open("models/class_names.json", "r") as f:
    class_names = json.load(f)
    reverse_map = {v: k for k, v in class_names.items()}

print("="*50)
print("Model Test")
print("="*50)
print(f"Classes: {reverse_map}")
print("="*50)

# Test with a basophil image
test_path = r"dataset_extracted2\PBC_dataset_normal_DIB\basophil"
if os.path.exists(test_path):
    # Get first image
    images = [f for f in os.listdir(test_path) if f.endswith('.jpg')]
    if images:
        img_path = os.path.join(test_path, images[0])
        print(f"\nTesting with: {img_path}")
        
        # Preprocess
        img = Image.open(img_path).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        predictions = model.predict(img_array)[0]
        top_idx = np.argmax(predictions)
        top_class = reverse_map[top_idx]
        confidence = predictions[top_idx] * 100
        
        print(f"\n🔬 Prediction: {top_class}")
        print(f"📊 Confidence: {confidence:.2f}%")
        print("\nAll predictions:")
        for idx, prob in enumerate(predictions):
            print(f"   {reverse_map[idx]}: {prob*100:.2f}%")