# back.py - Fixed version with proper class mapping
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io
import logging
import os
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="HemaToVision API")

# CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model and class names
model = None
class_names = None
reverse_class_map = None

try:
    import tensorflow as tf
    
    # Load the trained model
    if os.path.exists("models/blood_cell_classifier.h5"):
        model = tf.keras.models.load_model("models/blood_cell_classifier.h5")
        logger.info("✅ Loaded trained model")
        
        # Load the class names from training
        if os.path.exists("models/class_names.json"):
            with open("models/class_names.json", "r") as f:
                class_names = json.load(f)
                # class_names from training: {'basophil': 0, 'eosinophil': 1, ...}
                # Create reverse mapping: {0: 'basophil', 1: 'eosinophil', ...}
                reverse_class_map = {v: k for k, v in class_names.items()}
                logger.info(f"✅ Loaded class mapping: {reverse_class_map}")
        else:
            logger.warning("⚠️ class_names.json not found. Using default classes.")
    else:
        logger.warning("⚠️ No trained model found at models/blood_cell_classifier.h5")
        
except Exception as e:
    logger.warning(f"⚠️ Could not load model: {e}")

# Fallback classes if model not loaded
if not reverse_class_map:
    reverse_class_map = {
        0: "basophil",
        1: "eosinophil",
        2: "erythroblast",
        3: "immature_granulocytes",
        4: "lymphocyte",
        5: "monocyte",
        6: "neutrophil",
        7: "platelet"
    }

BLOOD_CELL_CLASSES = list(reverse_class_map.values())

def preprocess_image(image_bytes: bytes):
    """Preprocess image for model prediction (must match training preprocessing)"""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize((224, 224))
        # Same preprocessing as training: normalize to [0,1]
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array, image
    except Exception as e:
        logger.error(f"Image preprocessing error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

def extract_cellular_features(image: Image.Image) -> dict:
    """Extract morphological features from blood cell image"""
    img_np = np.array(image.convert('RGB'))
    mean_rgb = np.mean(img_np, axis=(0, 1))
    r_mean, g_mean, b_mean = mean_rgb
    r_g_ratio = r_mean / (g_mean + 1e-6)
    gray = np.mean(img_np, axis=2)
    edge_density = np.std(gray) / 255.0
    intensity_std = float(np.std(gray))
    
    features = {
        'red_green_ratio': round(float(r_g_ratio), 3),
        'edge_density': round(float(edge_density), 3),
        'intensity_std': round(intensity_std, 2)
    }
    return features

def heuristic_classify(features: dict) -> dict:
    """Fallback classification if model not available"""
    if features['edge_density'] > 0.25:
        top_class = "neutrophil"
        confidence = 65.0
    elif features['red_green_ratio'] > 1.3:
        top_class = "eosinophil"
        confidence = 70.0
    else:
        top_class = "lymphocyte"
        confidence = 60.0
    
    return {"class": top_class, "confidence": confidence}

@app.get("/")
async def root():
    return {
        "service": "HemaToVision API",
        "version": "2.1.0",
        "status": "running",
        "model_loaded": model is not None,
        "classes_loaded": len(reverse_class_map),
        "endpoints": ["/classify", "/health", "/classes", "/model-info"]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "HemaToVision",
        "model_loaded": model is not None,
        "classes": BLOOD_CELL_CLASSES
    }

@app.get("/model-info")
async def model_info():
    """Get information about the loaded model"""
    return {
        "model_loaded": model is not None,
        "num_classes": len(BLOOD_CELL_CLASSES),
        "classes": BLOOD_CELL_CLASSES,
        "class_mapping": reverse_class_map
    }

@app.get("/classes")
async def get_classes():
    return {"classes": BLOOD_CELL_CLASSES}

@app.post("/classify")
async def classify_blood_cell(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and preprocess image
        image_bytes = await file.read()
        processed_img, pil_image = preprocess_image(image_bytes)
        features = extract_cellular_features(pil_image)
        
        # Use trained model if available
        if model is not None:
            # Get predictions
            predictions = model.predict(processed_img, verbose=0)[0]
            
            # Get top prediction
            top_index = int(np.argmax(predictions))
            top_confidence = float(predictions[top_index]) * 100
            
            # Get class name from the reverse mapping (this ensures correct label!)
            top_class = reverse_class_map.get(top_index, "Unknown")
            
            mode = "deep_learning"
            
            # Get all probabilities with correct class names
            probabilities = {}
            for idx, prob in enumerate(predictions):
                class_name = reverse_class_map.get(idx, f"class_{idx}")
                probabilities[class_name] = float(prob) * 100
            
            logger.info(f"Prediction: {top_class} with {top_confidence:.1f}% confidence")
            
        else:
            # Fallback to heuristic
            result = heuristic_classify(features)
            top_class = result["class"]
            top_confidence = result["confidence"]
            mode = "heuristic"
            probabilities = {top_class: top_confidence}
        
        # Sort probabilities
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        analysis_notes = generate_analysis_notes(top_class, features, top_confidence)
        
        response = {
            "success": True,
            "prediction": {
                "class": top_class,
                "confidence_percent": round(top_confidence, 2)
            },
            "all_probabilities": {k: round(v, 2) for k, v in sorted_probs},
            "morphological_features": features,
            "analysis_notes": analysis_notes,
            "classification_mode": mode
        }
        
        return JSONResponse(content=response)
    
    except Exception as e:
        logger.error(f"Classification error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

def generate_analysis_notes(cell_class: str, features: dict, confidence: float) -> str:
    """Generate clinical analysis notes"""
    notes = []
    
    # Convert to lowercase for comparison
    cell_class_lower = cell_class.lower()
    
    if cell_class_lower == "neutrophil":
        notes.append("Segmented nucleus with 2-5 lobes - typical neutrophil morphology")
        notes.append("Primary defense against bacterial infections")
        if features['edge_density'] > 0.25:
            notes.append("Prominent nuclear segmentation observed")
    
    elif cell_class_lower == "eosinophil":
        notes.append("Bilobed nucleus with red/orange granules")
        notes.append("Associated with parasitic infections and allergies")
        if features['red_green_ratio'] > 1.2:
            notes.append("Eosinophilic granules clearly visible")
    
    elif cell_class_lower == "basophil":
        notes.append("Large dark purple granules")
        notes.append("Involved in allergic and inflammatory responses")
        notes.append("Rarest granulocyte in peripheral blood")
    
    elif cell_class_lower == "lymphocyte":
        notes.append("High nuclear-to-cytoplasmic ratio")
        notes.append("Key player in adaptive immunity")
    
    elif cell_class_lower == "monocyte":
        notes.append("Kidney-shaped or horseshoe nucleus")
        notes.append("Precursor to tissue macrophages")
        notes.append("Largest cell in peripheral blood")
    
    elif cell_class_lower == "platelet":
        notes.append("Small disc-shaped cell fragments")
        notes.append("Essential for blood clotting and hemostasis")
        if features.get('circularity', 0) > 0.85:
            notes.append("Normal round morphology")
    
    elif cell_class_lower == "erythroblast":
        notes.append("Nucleated red blood cell precursor")
        notes.append("Normally found in bone marrow only")
        notes.append("Peripheral presence may indicate pathology")
    
    elif "immature" in cell_class_lower:
        notes.append("Immature granulocyte detected")
        notes.append("Further evaluation recommended")
    
    else:
        notes.append(f"{cell_class} cell detected")
        notes.append("Clinical correlation recommended")
    
    if confidence > 85:
        notes.append("High confidence classification")
    elif confidence < 70:
        notes.append("Consider manual verification")
    
    return " | ".join(notes)

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("🩸 HemaToVision Backend Server v2.1")
    print("="*60)
    print(f"✅ Model Loaded: {model is not None}")
    if model:
        print(f"✅ Classes: {BLOOD_CELL_CLASSES}")
    print("📍 API URL: http://localhost:8000")
    print("📖 API Docs: http://localhost:8000/docs")
    print("🔍 Model Info: http://localhost:8000/model-info")
    print("\n⚠️  Press CTRL+C to stop the server")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
    #C:
    #cd C:\Users\DELL\Documents\hemetovision
    #D:\tf_env\Scripts\python.exe back.py