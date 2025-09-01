import os
import random
import tempfile
import urllib.request
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
from tqdm import tqdm
import math

# === Config ===
IMG_DIR = "Model/Images"
SCORES_CSV = "Model/Scores.csv"
MODEL_PATH = "Folds/seed_3516510858_fold1.pt"
TO_EVALUATE_DIR = "Model/ToEvaluate"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Model Definition ===
def build_model():
    weights = EfficientNet_B1_Weights.DEFAULT
    model = efficientnet_b1(weights=weights)
    model.classifier[0] = nn.Dropout(0.3)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    return model

# === Preprocessing ===
class ResizeWithPadding:
    def __init__(self, size=224, fill_color=(0, 0, 0)):
        self.size = size
        self.fill_color = fill_color

    def __call__(self, img):
        img_np = np.array(img)[:, :, ::-1]  # RGB to BGR
        h, w = img_np.shape[:2]
        scale = self.size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        new_img = np.full((self.size, self.size, 3), self.fill_color[::-1], dtype=np.uint8)
        x_offset = (self.size - new_w) // 2
        y_offset = (self.size - new_h) // 2
        new_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        return Image.fromarray(new_img[:, :, ::-1])

transform = transforms.Compose([
    ResizeWithPadding(size=224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === Load Model ===
def load_model():
    model = build_model().to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.eval()
    return model

# === Prediction ===
def predict_image(model, image):
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        prediction = model(tensor).item()
    return max(1.0, min(5.0, prediction))

# === Evaluate Entire Dataset with Metrics ===
def evaluate_model_on_dataset():
    data = pd.read_csv(SCORES_CSV, header=None, names=["filename", "score"])
    model = load_model()
    preds = []
    trues = []

    for _, row in tqdm(data.iterrows(), total=len(data), desc="Evaluating"):
        img_path = os.path.join(IMG_DIR, row['filename'])
        if not os.path.exists(img_path):
            continue
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            continue
        pred = predict_image(model, image)
        true = float(row["score"])

        preds.append(pred)
        trues.append(true)

    if preds:
        mae = mean_absolute_error(trues, preds)
        rmse = math.sqrt(mean_squared_error(trues, preds))
        pearson, _ = pearsonr(trues, preds)
        acc = sum(abs(p - t) <= 0.5 for p, t in zip(preds, trues)) / len(preds)

        print("\n================== Evaluation Metrics ==================")
        print(f"Total Samples Evaluated: {len(preds)}")
        print(f"MAE     : {mae:.4f}")
        print(f"RMSE    : {rmse:.4f}")
        print(f"Pearson : {pearson:.4f}")
        print(f"Accuracy (<= 0.5 diff): {acc * 100:.2f}%")
        print("========================================================")
    else:
        print("No valid predictions were made.")

# === Entry Point ===
if __name__ == "__main__":
    evaluate_model_on_dataset()
