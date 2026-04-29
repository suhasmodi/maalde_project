import os
import pickle
import ssl
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from flask import Flask, render_template, request, jsonify

# Bypass SSL for weights download if needed
ssl._create_default_https_context = ssl._create_unverified_context

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16 MB max

MODEL_PATH = "demand_model.pkl"

# --- Global Variables for Models ---
xgboost_model = None
resnet_model = None
preprocess_transform = None

def load_models():
    global xgboost_model, resnet_model, preprocess_transform
    print("Loading models into memory...")
    
    # Load XGBoost
    try:
        with open(MODEL_PATH, "rb") as f:
            xgboost_model = pickle.load(f)
    except FileNotFoundError:
        print(f"Warning: {MODEL_PATH} not found. Ensure train.py has been run.")
        xgboost_model = None

    # Load ResNet18 Feature Extractor
    weights = ResNet18_Weights.DEFAULT
    resnet_model = resnet18(weights=weights)
    resnet_model.fc = torch.nn.Identity()
    resnet_model.eval()
    
    preprocess_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=weights.transforms().mean, std=weights.transforms().std),
    ])

# Load models at startup
load_models()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def predict_demand():
    if xgboost_model is None:
        return jsonify({"error": "ML Model not loaded on server."}), 500
        
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400
        
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file."}), 400
        
    try:
        rate_str = request.form.get("rate")
        rate = float(rate_str)
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid or missing 'rate' value."}), 400

    try:
        # Extract features
        image = Image.open(file.stream).convert("RGB")
        input_tensor = preprocess_transform(image)
        input_batch = input_tensor.unsqueeze(0)

        with torch.no_grad():
            output = resnet_model(input_batch)
        img_features = output.numpy().flatten()
        
        # Predict
        X_input = np.hstack(([rate], img_features)).reshape(1, -1)
        predicted_qty = xgboost_model.predict(X_input)[0]
        
        return jsonify({
            "success": True,
            "predicted_qty": max(0, int(round(predicted_qty)))
        })
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
