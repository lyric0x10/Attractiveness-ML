import os
import random
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from tqdm import tqdm

# === Config ===
CSV_PATH = "Model/Scores.csv"
IMG_DIR = "Model/Images"
CACHE_DIR = "Model/Cache"
MODEL_SAVE_DIR = "Folds"
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
EPOCHS = 35
NUM_FOLDS = 5
SEED = random.randint(0, 2**32 - 1)
NUM_WORKERS = 0
PATIENCE = 10
GRADIENT_CLIP = 1.0
MODEL_SAVE_PREFIX = f"seed_{SEED}_"

# === Set seeds for reproducibility ===
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# === Dataset ===
class CachedImageDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        os.makedirs(CACHE_DIR, exist_ok=True)

        self.pil_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(0.05, 0.05, 0.05, 0.02),  # Less intense jitter
            transforms.RandomPerspective(distortion_scale=0.075, p=1.0),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
            transforms.RandomResizedCrop(224, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
        ])


        self.tensor_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_name = row.get('File') or row.get('file') or row.get('filename')
        if not image_name:
            raise ValueError("Expected a column like 'File' or 'file' in the CSV")

        image_path = os.path.join(IMG_DIR, image_name)
        cache_path = os.path.join(CACHE_DIR, f"{os.path.basename(image_path)}.pt")

        if os.path.exists(cache_path):
            img_tensor = torch.load(cache_path).float() / 255.0
        else:
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, _ = img.shape
            scale = 224 / max(h, w)
            resized = cv2.resize(img, (int(w * scale), int(h * scale)))
            canvas = np.zeros((224, 224, 3), dtype=np.uint8)
            y_off = (224 - resized.shape[0]) // 2
            x_off = (224 - resized.shape[1]) // 2
            canvas[y_off:y_off + resized.shape[0], x_off:x_off + resized.shape[1]] = resized
            img_tensor = torch.from_numpy(canvas).permute(2, 0, 1)
            torch.save(img_tensor, cache_path)

        img_pil = transforms.ToPILImage()(img_tensor)
        img_pil = self.pil_transform(img_pil)
        img_tensor = self.tensor_transform(img_pil)

        score = float(row['Score'])
        score = max(1.0, min(5.0, score))  # clamp to [1, 5]

        return img_tensor, torch.tensor(score, dtype=torch.float32)

# === Model ===
def build_model():
    weights = EfficientNet_B1_Weights.DEFAULT
    model = efficientnet_b1(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    return model


# === Training Utilities ===
def combined_loss(pred, target):
    return 0.7 * nn.functional.l1_loss(pred, target) + 0.3 * nn.functional.mse_loss(pred, target)


def train_fold(train_df, val_df, fold):
    train_ds = CachedImageDataset(train_df)
    val_ds = CachedImageDataset(val_df)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = build_model().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    criterion = combined_loss

    best_mae = float('inf')
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for x, y in tqdm(train_dl, desc=f"[Fold {fold}] Epoch {epoch:03d} - Training"):
            x, y = x.cuda(), y.cuda().unsqueeze(1)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        model.eval()
        val_loss, val_mae = 0, 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.cuda(), y.cuda().unsqueeze(1)
                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item() * x.size(0)
                val_mae += torch.abs(pred - y).sum().item()

        val_loss /= len(val_ds)
        val_mae /= len(val_ds)
        print(f"[Fold {fold}] Epoch {epoch:03d} | Train Loss: {total_loss / len(train_ds):.4f} | Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.4f}")

        # Optional: debug output range
        if epoch == 1:
            sample_x, _ = next(iter(train_dl))
            sample_pred = model(sample_x.cuda())
            print(f"[Debug] Prediction range: {sample_pred.min().item():.2f} to {sample_pred.max().item():.2f}")

        scheduler.step(val_loss)

        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, f"{MODEL_SAVE_PREFIX}fold{fold}.pt"))
            print(f"[+] Saved model with MAE {best_mae:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("[!] Early stopping due to no improvement.")
                break

    return best_mae

# === Main ===
def main():
    df = pd.read_csv(CSV_PATH)
    assert 'Score' in df.columns, "CSV must have a 'Score' column"

    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        print(f"\n[>>] Starting fold {fold + 1}")
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        best_mae = train_fold(train_df, val_df, fold + 1)

if __name__ == "__main__":
    main()
