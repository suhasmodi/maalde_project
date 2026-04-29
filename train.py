import os
import random
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

DATA_FILE = "AI ML Task Sheet - sales data.csv"
IMAGE_DIR = "1"
MODEL_PATH = "demand_model.pkl"

print("Loading sales data...")
df = pd.read_csv(DATA_FILE)

# Aggregate data by product code
# We predict the total quantity sold based on the average rate (price) and image
agg_df = df.groupby("code").agg({
    "qty": "sum",       # Total sales quantity (Target)
    "rate": "mean"      # Average rate (Feature)
}).reset_index()

print(f"Total unique products in sales data: {len(agg_df)}")

# Since there is no explicit mapping between product codes and the provided images,
# we will randomly assign available images to the product codes for this demonstration.
print("Mapping images to product codes...")
available_images = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpeg', '.jpg', '.png'))]

if not available_images:
    raise FileNotFoundError(f"No images found in the '{IMAGE_DIR}' directory.")

random.seed(42)
code_to_image = {}
for code in agg_df['code'].unique():
    code_to_image[code] = os.path.join(IMAGE_DIR, random.choice(available_images))

agg_df['image_path'] = agg_df['code'].map(code_to_image)

# --- 4. IMAGE FEATURE EXTRACTION (ResNet18) ---
print("Initializing Image Feature Extractor (ResNet18)...")
# Load a pre-trained ResNet model and remove the final classification layer
weights = ResNet18_Weights.DEFAULT
resnet = resnet18(weights=weights)
resnet.fc = torch.nn.Identity() # Remove final fully connected layer
resnet.eval()

# Standard image transforms for ResNet
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=weights.transforms().mean, std=weights.transforms().std),
])

def extract_image_features(img_path):
    try:
        image = Image.open(img_path).convert("RGB")
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0) 

        with torch.no_grad():
            output = resnet(input_batch)
        return output.numpy().flatten()
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return np.zeros(512) # ResNet18 outputs 512-dim vector

print("Extracting features from images (this may take a minute)...")
agg_df['image_features'] = agg_df['image_path'].apply(extract_image_features)

# Combine features: [Rate] + [Image Features (512 dims)]
print("Preparing dataset for training...")
X_rate = agg_df[['rate']].values
X_image = np.vstack(agg_df['image_features'].values)
X = np.hstack((X_rate, X_image)) # Final feature matrix
y = agg_df['qty'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training XGBoost Regressor...")


model=RandomForestRegressor(
    n_estimators=700,
    max_depth=8,
    random_state=42
)


model.fit(X_train, y_train)

# --- 7. EVALUATION ---
print("Evaluating model...")
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)

print(f"--- Evaluation Results ---")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# --- 8. SAVE MODEL ---
print(f"Saving trained model to {MODEL_PATH}...")
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print("Training pipeline completed successfully.")
