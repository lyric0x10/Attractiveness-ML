import os
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import cv2

# === Config ===
IMG_DIR = "Model/Images"
SCORES_CSV = "Model/Scores.csv"
FOLDER_PATH = "Folds"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_COUNT = 5
SAMPLE_SIZE = 75

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

# === Model Definition ===
def build_model():
    weights = EfficientNet_B1_Weights.DEFAULT
    model = efficientnet_b1(weights=weights)
    model.classifier[0] = nn.Dropout(0.3)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    return model

# === Prediction ===
def load_model(MODEL_PATH):
    print(MODEL_PATH)
    model = build_model().to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.eval()
    return model

def predict_image(model, image):
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        prediction = model(tensor).item()
        return max(1.0, min(5.0, prediction))  # Clamp output like training

def get_sample(df, sample_size):
    return df.sample(n=min(sample_size, len(df)), random_state=42).reset_index(drop=True)

def evaluate_sample(sample, model):
    total_distance = 0.0
    correct = 0
    valid = 0
    for _, row in sample.iterrows():
        img_path = os.path.join(IMG_DIR, row['File'])
        if not os.path.exists(img_path):
            continue
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            continue
        pred = predict_image(model, img)
        true = float(row['Score'])
        diff = abs(pred - true)
        total_distance += diff
        valid += 1
        if diff <= 0.5:
            correct += 1
    if valid == 0:
        return None
    return {
        'average_mae': total_distance / valid,
        'accuracy_within_0.5': 100 * correct / valid
    }

def print_results(all_results):
    print("\n" + "=" * 50)
    print("FINAL EVALUATION RESULTS".center(50))
    print("=" * 50)

    all_maes = []
    all_accuracies = []
    model_perf = {}

    for fold_result in all_results:
        sample_num = fold_result['sample_number']
        print(f"\nSample {sample_num} Results:")
        print("-" * 45)
        for res in fold_result['results']:
            name = res['name']
            mae = res['average_mae']
            acc = res['accuracy_within_0.5']
            print(f"Model: {name:<20} | MAE: {mae:.4f} | Accuracy (<=0.5): {acc:.2f}%")
            all_maes.append(mae)
            all_accuracies.append(acc)
            if name not in model_perf:
                model_perf[name] = {'maes': [], 'accs': []}
            model_perf[name]['maes'].append(mae)
            model_perf[name]['accs'].append(acc)

    avg_mae = sum(all_maes) / len(all_maes)
    avg_acc = sum(all_accuracies) / len(all_accuracies)

    best_model = max(model_perf.items(), key=lambda x: sum(x[1]['accs']) / len(x[1]['accs']))[0]
    best_acc = sum(model_perf[best_model]['accs']) / len(model_perf[best_model]['accs'])

    print("\n" + "=" * 50)
    print("SUMMARY STATISTICS".center(50))
    print("=" * 50)
    print(f"\nAverage MAE: {avg_mae:.4f}")
    print(f"Average Accuracy (<=0.5): {avg_acc:.2f}%")
    print(f"Best Model: {best_model} with {best_acc:.2f}% accuracy")

    print("\n" + "=" * 50)
    print("MODEL RANKINGS BY ACCURACY".center(50))
    print("=" * 50)
    ranked = sorted(
        [(name, sum(perf['accs']) / len(perf['accs'])) for name, perf in model_perf.items()],
        key=lambda x: x[1], reverse=True
    )
    for i, (name, acc) in enumerate(ranked, 1):
        print(f"{i}. {name:<20} {acc:.2f}%")

# === Entry Point ===
if __name__ == "__main__":
    df = pd.read_csv(SCORES_CSV)
    all_results = []

    for i in range(SAMPLE_COUNT):
        sample = get_sample(df, SAMPLE_SIZE)
        model_files = [f for f in os.listdir(FOLDER_PATH) if f.endswith(".pt")]
        sample_results = []
        for m in model_files:
            path = os.path.join(FOLDER_PATH, m)
            model = load_model(path)
            result = evaluate_sample(sample, model)
            if result:
                result['name'] = m
                sample_results.append(result)
        all_results.append({"sample_number": i + 1, "results": sample_results})

    print_results(all_results)