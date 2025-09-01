import os
import csv
import glob
import cv2
import numpy as np
from PIL import Image
from math import atan2, degrees

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models.efficientnet import efficientnet_b1, EfficientNet_B1_Weights
from sklearn.metrics import mean_squared_error, mean_absolute_error

WEIGHTS_PATH = "Folds/seed_3516510858_fold3.pt"
INPUT_PATH = "Model/ToEvaluate"
CSV_PATH = "Model/Scores.csv"
MODEL_INPUT_SIZE = 240

class ResizeWithPadding:
    def __init__(self, size=MODEL_INPUT_SIZE, fill_color=(0, 0, 0)):
        self.size = int(size)
        self.fill = fill_color
    def __call__(self, img):
        w, h = img.size
        scale = self.size / max(w, h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        img_r = img.resize((nw, nh), Image.Resampling.LANCZOS)
        bg = Image.new("RGB", (self.size, self.size), self.fill)
        bg.paste(img_r, ((self.size - nw) // 2, (self.size - nh) // 2))
        return bg

transform = transforms.Compose([
    ResizeWithPadding(MODEL_INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])

def build_model(device):
    weights = EfficientNet_B1_Weights.DEFAULT
    model = efficientnet_b1(weights=weights)
    model.classifier[0] = nn.Dropout(0.3)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    return model.to(device)

def clamp(v, low, high):
    return max(low, min(high, v))

def predict_image(path, model, device):
    img = Image.open(path).convert("RGB")
    t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(t).squeeze().float().item()
    return float(clamp(out, 0.0, 5.0))

def gather_images(input_path):
    if os.path.isdir(input_path):
        files = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp"):
            files.extend(glob.glob(os.path.join(input_path, ext)))
        return sorted(files)
    elif os.path.isfile(input_path):
        return [input_path]
    return []

def load_ground_truth(csv_path):
    if not os.path.exists(csv_path):
        return {}
    truth = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                truth[r["filename"]] = float(r["rating"])
            except:
                continue
    return truth

def main(INPUT_PATH):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(device)
    if os.path.isfile(WEIGHTS_PATH):
        state = torch.load(WEIGHTS_PATH, map_location=device)
        model.load_state_dict(state, strict=False)
    model.eval()

    truth = load_ground_truth(CSV_PATH)

    preds, gts = [], []
    score = predict_image(INPUT_PATH, model, device)
    score_10 = score * 2
    print(f"{os.path.basename(INPUT_PATH)} -> {score:.4f} (1-5) | {score_10:.4f} (1-10)")
    preds.append(score)
    if os.path.basename(INPUT_PATH) in truth:
        gts.append(truth[os.path.basename(INPUT_PATH)])

    if gts and len(gts) == len(preds):
        rmse = mean_squared_error(gts, preds, squared=False)
        mae = mean_absolute_error(gts, preds)
        print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}")

if __name__ == "__main__":
    main(r"chico.png")
