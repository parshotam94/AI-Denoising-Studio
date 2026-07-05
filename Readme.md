# AI Image Denoising Studio (MNIST Autoencoder)

An elegant, production-ready AI SaaS web application that performs high-fidelity image denoising on handwritten digits using a Convolutional Neural Network (CNN) Autoencoder. Built with a modular FastAPI backend and an animated, responsive, dark-themed Glassmorphism frontend UI.

---

## 🚀 Features

- **Production SaaS UI:** Designed with an immersive slate-dark theme (`#0F172A`) featuring glassmorphism elements, custom form widgets, and sharp grid layouts.
- **Interactive Execution Canvas:** Select target digits (0–9) and dynamically inject structural Gaussian Noise through real-time slider matrices.
- **Asynchronous Processing Pipeline Animation:** Follow each visual framework step live as the system runs through *Dataset Matrix Extraction → Noise Injection → Compression → Latent Space Mapping → Deconvolution → Image Recovery & Assessment*.
- **Comprehensive Image Diagnostics:** Displays high-precision mathematical metric scores for Mean Square Error (**MSE**), Peak Signal-to-Noise Ratio (**PSNR**), and Structural Similarity Index (**SSIM**).
- **Secondary Classification Confidence:** Uses a pre-trained, auxiliary CNN Classifier to evaluate the legible structural validity of the restored frame.
- **Deep Architecture Profiler:** Interactive topology tree inspects individual model layers and shapes natively extracted from saved model files.
- **Sandbox Training Simulator:** Visual logging console streaming simulated epoch loops, optimization step data, and dynamic loss metrics tracking.

---

## 📁 Project Architecture

```text
ai_denoising_studio/
│
├── api/
│   ├── main.py             # FastAPI App Engine & Inference APIs
│   └── train_model.py       # Weights Compilation & Asset Pipeline
│
├── frontend/
│   ├── templates/
│   │   ├── index.html       # SaaS Landing Showcase
│   │   ├── dashboard.html   # Main Core Execution Canvas
│   │   ├── training.html    # Model Loop Monitor Simulator
│   │   └── visualization.html# Deep Architecture Profiler Matrix
│   └── static/
│       ├── css/
│       │   └── style.css    # Layout Styles & Keyframe Animations
│       └── js/
│           └── app.js       # Asynchronous Application Orchestrator
│
└── models/
│   ├── autoencoder.h5       # Saved Encoder/Decoder Weights (Generated)
│   ├── classifier.h5        # Saved Predictive Classifier (Generated)
│   ├── x_test.npy           # Normalized Inference Reference Data (Generated)
│   └── y_test.npy           # Image Ground-Truth Labels (Generated)
│
├── requirements.txt
```

## 🛠️ Technical Implementation Spec

### Machine Learning Models
- **CNN Autoencoder:** Composed of an entry **Encoder** utilizing contracting block combinations of `Conv2D` and `MaxPooling2D` to yield an ultra-compact `[7, 7, 16]` Latent Space Bottleneck vector, mirrored precisely across an expansive **Decoder** sequence of `UpSampling2D` and `Conv2D` layers running a boundary-clamping `sigmoid` activation out.
- **MNIST Classifier:** Lightweight auxiliary network featuring a localized convolution block, flattening layer, dense layer array, and an explicit 10-node `softmax` output head to evaluate predicted numbers.

### Performance Analytics Engine
- **MSE:** Calculated via scalar matrix squared differences: 
  $$\text{MSE} = \frac{1}{mn}\sum_{i=0}^{m-1}\sum_{j=0}^{n-1}[I(i,j) - K(i,j)]^2$$
- **PSNR:** Computed natively across raw spatial 8-bit conversions using the standard peak signal maximum ($255$).
- **SSIM:** Structured luminance, contrast, and structural comparison evaluations pulled from the structural image processing utility library `scikit-image`.

---

## ⚡ Quick Start

### 1. Environment Configurations
Clone or place the codebase files directly inside an active project folder directory and set up core operational dependencies:

```bash
pip install -r requirements.txt
```

### 2. Run Weights Preparation Sequences

Before booting up the service server, you must run the background compilation script *once* to populate dataset subsets and build localized network weights:

```bash
python api/train_model.py

```

*Note: This script automatically loads the MNIST data repository, structures optimization arrays, trains models across lightweight initial epochs, and populates artifacts inside the `/models` directory.*

### 3. Initialize ASGI Production Server Instance

Launch the application microservice utilizing the standard high-performance ASGI `uvicorn` engine execution framework:

```bash
uvicorn api.main:app --reload --port 8000

```

Once initialized, navigate your active browser client engine directly to:
👉 **`http://localhost:8000`**

---

## 📡 API Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| **GET** | `/` | Renders the primary SaaS landing interface viewport. |
| **GET** | `/dashboard` | Renders the real-time execution canvas and processing pipeline. |
| **GET** | `/training` | Renders the core training tracking console dashboard. |
| **GET** | `/visualization` | Renders the deep network topology profiler window. |
| **GET** | `/api/random-image/{digit}` | Pulls a random vector representation matrix filtered by a given integer. |
| **POST** | `/api/predict` | Accepts standard image constraints payload, processes the frame pipeline, and calculates mathematical metrics. |
| **GET** | `/api/model-info` | Returns the structural layout specifications for active model layers. |

```

```
