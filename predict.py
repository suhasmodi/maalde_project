import argparse
import pickle
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights

MODEL_PATH = "demand_model.pkl"

def get_image_extractor():
    weights = ResNet18_Weights.DEFAULT
    resnet = resnet18(weights=weights)
    resnet.fc = torch.nn.Identity()
    resnet.eval()
    
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=weights.transforms().mean, std=weights.transforms().std),
    ])
    return resnet, preprocess

def extract_features(img_path, resnet, preprocess):
    image = Image.open(img_path).convert("RGB")
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = resnet(input_batch)
    return output.numpy().flatten()

def predict(img_path, rate):
    # Load trained XGBoost model
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Model file '{MODEL_PATH}' not found. Please run train.py first.")
        return

    # Initialize feature extractor
    resnet, preprocess = get_image_extractor()
    
    # Extract image features
    try:
        img_features = extract_features(img_path, resnet, preprocess)
    except Exception as e:
        print(f"Error processing image '{img_path}': {e}")
        return

    # Prepare input feature vector: [rate, img_feature_1, ..., img_feature_512]
    X_input = np.hstack(([rate], img_features)).reshape(1, -1)
    
    # Predict
    predicted_qty = model.predict(X_input)[0]
    
    print("-" * 40)
    print("PREDICTION RESULT")
    print("-" * 40)
    print(f"Image: {img_path}")
    print(f"Proposed Rate: {rate}")
    print(f"Predicted Sales Quantity: {max(0, int(round(predicted_qty)))} units")
    print("-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict demand quantity for a new design.")
    parser.add_argument("--image", type=str, required=True, help="Path to the new design image.")
    parser.add_argument("--rate", type=float, required=True, help="Proposed price (rate) for the new design.")
    
    args = parser.parse_args()
    predict(args.image, args.rate)
