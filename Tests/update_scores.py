import os
import torch
import pandas as pd
from PIL import Image, ImageOps
from torchvision import transforms
import torch.nn as nn
import torchvision.models as models

class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()
        self.base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.base.fc = nn.Linear(self.base.fc.in_features, 1)

    def forward(self, x):
        return self.base(x)

def resize_with_padding(img, size=224):
    img.thumbnail((size, size), Image.Resampling.LANCZOS)
    delta_w = size - img.size[0]
    delta_h = size - img.size[1]
    padding = (delta_w // 2, delta_h // 2, delta_w - delta_w // 2, delta_h - delta_h // 2)
    return ImageOps.expand(img, padding, fill=0)

def clamp(min_val, max_val, value):
    return max(min_val, min(max_val, value))

# -- Config --
IMG_DIR = "Model/Images"
INPUT_CSV = "Model/Scores.csv"
SCALED_CSV = "Model/Scores_scaled.csv"
MODEL_PATH = "model_best.pth"

# -- Range conversion config --
CURRENT_RANGE = 3  # Original model was trained on 1–3 scale
NEW_RANGE = 5     # New desired scale

# -- Load image list --
original_data = pd.read_csv(INPUT_CSV, header=None, names=["filename", "score"])

transform = transforms.Compose([
    transforms.Lambda(lambda img: resize_with_padding(img, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -- Load model --
model = Regressor()
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

# -- Predict and scale --
scaled_results = []
total_distance = 0.0
num_samples = 0
correct = 0

for index, row in original_data.iterrows():
    img_path = os.path.join(IMG_DIR, row['filename'])
    true_score = float(row['score'])
    if not os.path.exists(img_path):
        print(f"Missing image: {img_path}")
        continue

    image = Image.open(img_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        predicted_score = model(input_tensor).item()

    # Scale prediction
    scaled_prediction = clamp(1, 5, round((predicted_score / CURRENT_RANGE) * NEW_RANGE))
    scaled_results.append([row['filename'], scaled_prediction])

    # Evaluation metrics (optional)
    distance = abs(predicted_score - true_score)
    total_distance += distance
    num_samples += 1
    if round(predicted_score) == round(true_score):
        correct += 1
    print(f"{row['filename']}: Pred = {predicted_score:.2f}, Scaled = {scaled_prediction:.2f}, True = {true_score:.2f}, Diff = {distance:.2f}")

# -- Save predictions to new CSV --
pd.DataFrame(scaled_results).to_csv(SCALED_CSV, index=False, header=False)

# -- Report --
if num_samples > 0:
    avg_distance = total_distance / num_samples
    print(f"\nAverage Distance: {avg_distance:.4f}")
    print(f"Rounded Accuracy: {round(correct / num_samples, 4) * 100:.2f}%")
else:
    print("No valid samples found.")
