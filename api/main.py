import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
import os
import time
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from skimage.metrics import structural_similarity as ssim

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(title="AI Image Denoising Studio")

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for frontend
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "frontend/static")), name="static")

# Global Variables for Models and Data
AUTOENCODER = None
CLASSIFIER = None
X_TEST = None
Y_TEST = None

@app.on_event("startup")
def load_assets():
    global AUTOENCODER, CLASSIFIER, X_TEST, Y_TEST
    try:
        AUTOENCODER = tf.keras.models.load_model('models/autoencoder.h5')
        CLASSIFIER = tf.keras.models.load_model('models/classifier.h5')
        X_TEST = np.load('models/x_test.npy')
        Y_TEST = np.load('models/y_test.npy')
    except Exception as e:
        print(f"Warning: Models or test data not loaded. Please run backend/train_model.py first. Error: {e}")

class PredictRequest(BaseModel):
    digit: int
    noise_level: float

def read_html(path: str) -> str:
    # Safely point lookup arrays directly to root folder allocations
    target_path = os.path.join(BASE_DIR, "frontend/templates", path)
    with open(target_path, "r", encoding="utf-8") as f:
        return f.read()

# HTML Pages Endpoints
@app.get("/", response_class=HTMLResponse)
def read_home(): return read_html("index.html")

@app.get("/dashboard", response_class=HTMLResponse)
def read_dashboard(): return read_html("dashboard.html")

@app.get("/training", response_class=HTMLResponse)
def read_training(): return read_html("training.html")

@app.get("/visualization", response_class=HTMLResponse)
def read_visualization(): return read_html("visualization.html")

# API Logic Endpoints
@app.get("/api/random-image/{digit}")
def get_random_image(digit: int):
    if X_TEST is None or Y_TEST is None:
        raise HTTPException(status_code=500, detail="Models/Data not loaded.")
    indices = np.where(Y_TEST == digit)[0]
    if len(indices) == 0:
        raise HTTPException(status_code=404, detail="Digit not found.")
    idx = int(np.random.choice(indices))
    img = X_TEST[idx].reshape(28, 28).tolist()
    return {"digit": digit, "image": img}

@app.post("/api/predict")
def predict(data: PredictRequest):
    if AUTOENCODER is None or CLASSIFIER is None:
        raise HTTPException(status_code=500, detail="Models not loaded.")
        
    # 1. Fetch matching random original image
    indices = np.where(Y_TEST == data.digit)[0]
    idx = int(np.random.choice(indices))
    orig_img = X_TEST[idx] # Shape: (28, 28, 1)

    # 2. Inject Noise
    start_time = time.time()
    noise = np.random.normal(loc=0.0, scale=data.noise_level, size=orig_img.shape)
    noisy_img = np.clip(orig_img + noise, 0.0, 1.0)

    # 3. Process through Autoencoder
    input_batch = np.expand_dims(noisy_img, axis=0)
    denoised_batch = AUTOENCODER.predict(input_batch, verbose=0)
    denoised_img = denoised_batch[0]

    # 4. Process Classification Confidence
    class_pred = CLASSIFIER.predict(denoised_batch, verbose=0)[0]
    pred_digit = int(np.argmax(class_pred))
    confidence = float(class_pred[pred_digit])
    inference_time = (time.time() - start_time) * 1000 # MS

    # 5. Compute performance comparisons
    orig_flat = (orig_img.squeeze() * 255).astype(np.uint8)
    denoised_flat = (denoised_img.squeeze() * 255).astype(np.uint8)
    
    mse_val = float(np.mean((orig_img - denoised_img) ** 2))
    psnr_val = float(cv2.PSNR(orig_flat, denoised_flat))
    ssim_val = float(ssim(orig_flat, denoised_flat))

    return {
        "original": orig_img.squeeze().tolist(),
        "noisy": noisy_img.squeeze().tolist(),
        "denoised": denoised_img.squeeze().tolist(),
        "metrics": {
            "mse": round(mse_val, 5),
            "psnr": round(psnr_val, 2),
            "ssim": round(ssim_val, 4),
            "inference_time_ms": round(inference_time, 2)
        },
        "classification": {
            "predicted_digit": pred_digit,
            "confidence": round(confidence * 100, 2)
        }
    }

@app.get("/api/model-info")
def get_model_info():
    return {
        "encoder_layers": ["Input(28,28,1)", "Conv2D(32, Relu)", "MaxPooling2D", "Conv2D(16, Relu)", "MaxPooling2D"],
        "latent_space_shape": [7, 7, 16],
        "decoder_layers": ["Conv2D(16, Relu)", "UpSampling2D", "Conv2D(32, Relu)", "UpSampling2D", "Conv2D(1, Sigmoid)"]
    }