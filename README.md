# Attractiveness AI — README

A complete, local **face-attractiveness rating** toolkit (labeling GUI + training pipeline) built in Python + PyTorch.
This README is tailored to the code in this repository (labeling GUI: `label.py`, training: `train_model.py`) and explains how to run, train, evaluate, and contribute. See the referenced implementation for details. &#x20;

---

## ✨ Project Summary

This repository implements:

* a **labeling / rating GUI** (`FaceRater`) for quickly collecting human labels or reviewing AI-generated faces (uses MediaPipe face landmarks + a small UI).&#x20;
* an **EfficientNet-B1 regression model** (single-output) trained to predict attractiveness scores from face crops, with a K-Fold training pipeline and caching for images.&#x20;

Outputs and artifacts:

* Processed images stored under `Model/Images` and a labeling CSV `Model/Scores.csv`.&#x20;
* Trained fold weights saved to `Folds/` (e.g. `seed_<n>_fold1.pt`).&#x20;

---

## 🔧 Requirements

Minimum environment:

* Python 3.8+
* PyTorch (CUDA if available)
* torchvision
* opencv-python
* mediapipe
* pillow
* pandas
* scikit-learn
* tqdm
* customtkinter (for the GUI)
* other standard libs (numpy, csv, urllib, etc.)

Install (example):

```bash
python -m pip install torch torchvision opencv-python mediapipe pillow pandas scikit-learn tqdm customtkinter
```

> Use the correct PyTorch wheel for your CUDA version where applicable.

---

## 📁 Repo layout (important files)

* `label.py` — labeling UI, image processing, local inference/prediction code.&#x20;
* `train_model.py` — dataset, augmentations, K-Fold training loop, model save & early stopping.&#x20;
* `Model/Images/` — processed images used for training.&#x20;
* `Model/ToEvaluate/` — images you want to rate in the GUI.&#x20;
* `Model/Scores.csv` — CSV of labeled examples (filename, score).&#x20;
* `Folds/` — saved model weights after training folds.&#x20;

---

## ▶ Quickstart — Labeling (GUI)

1. Prepare images you want to label. Put them into `Model/ToEvaluate/` (jpg/png).
2. Run the GUI:

```bash
python label.py
```

What it does (implementation notes):

* Loads/initializes an EfficientNet-B1 model and a saved fold checkpoint for quick predictions.&#x20;
* Uses MediaPipe FaceMesh to detect landmarks, rotate and crop faces for consistent model input. Processed crops are saved to `Model/Images/`.&#x20;
* The GUI shows a preview, a predicted score, and buttons for rating (1–5). Labeled rows are appended to `Model/Scores.csv`.&#x20;

Shortcuts:

* Keys `1`..`5` — rate
* `s` — skip
* `d` — delete current image

---

## ▶ Quickstart — Training

1. Make sure `Model/Scores.csv` exists and contains a `Score` column with rows mapping image filenames (relative to `Model/Images/`) to numeric scores (1–5).&#x20;
2. Optionally prepare cached images (`Model/Cache/`) — the training script will generate caches automatically.
3. Run training:

```bash
python train_model.py
```

Key training details (from `train_model.py`):

* Uses EfficientNet-B1 (pretrained weights) with a final single-output linear regressor.&#x20;
* Combined loss: `0.7 * L1 + 0.3 * MSE`.&#x20;
* K-Fold (default 5 folds) cross-validation; saves best fold weights to `Folds/`.&#x20;
* Hyperparameters (defaults in the script): `batch_size=64`, `epochs=35`, `lr=1e-4`, `patience=10` for early stopping.&#x20;

---

## 🧭 Data format & preprocessing

* CSV (`Model/Scores.csv`) must contain at least a column named `Score` and a filename column (script accepts `File`, `file`, `filename` or `filename`-like columns). Each row links the image file in `Model/Images/` to a numeric score.&#x20;
* Image preprocessing:

  * Face detection + alignment via MediaPipe (face landmarks), cropping to the facial bounding box with small padding, rotation correction.&#x20;
  * Training pipeline applies augmentation (flip, rotation, color jitter, perspective, gaussian blur, random resized crop) and then center/canvas resize to 224×224 for EfficientNet.&#x20;

---

## 🔬 Model & Training specifics

* **Architecture**: EfficientNet-B1 (pretrained) with final classifier replaced by `Linear(in_features, 1)`. Dropout applied in the GUI model variant. &#x20;
* **Objective**: regression to a 1–5 attractiveness score (clamped). The training loss mixes L1 and MSE to balance robustness and squared error penalization.&#x20;
* **Evaluation**: validation MAE is printed each epoch. Best weights per fold are saved. Early stopping is used to avoid overfitting.&#x20;

---

## ⚠️ Ethics, limitations & responsible use

This is an inherently **sensitive and potentially harmful** application. Ratings of attractiveness are subjective and can encode, amplify, or weaponize biases (race, gender, age, photography style, etc.). Before using or sharing results:

* **Do not deploy** this system for hiring, law enforcement, credit decisions, or any consequential decision-making.
* Be transparent: collect consent for any human images, and document how labels were collected.&#x20;
* Expect **bias**: dataset composition, labeler demographics and UI labeling choices will strongly influence predictions. Validate on balanced datasets and measure per-group error rates.&#x20;
* Consider alternatives: focus on non-personal tasks, use anonymized or synthetic data, or avoid modeling attractiveness entirely if you cannot control for fairness and consent.

If you plan to publish or demonstrate, include an ethics statement, dataset provenance, and limitations in the write-up.

---

## ✅ Tips & troubleshooting

* If MediaPipe fails to detect a face, `label.py` falls back to a thumbnail without alignment — watch the console for messages.&#x20;
* If training runs out of memory, reduce `BATCH_SIZE` or set `NUM_WORKERS=0`. The script defaults `NUM_WORKERS=0`.&#x20;
* Model checkpoints are saved as `seed_<n>_fold{fold}.pt` — keep these to run inference in the GUI or for ensemble predictions.&#x20;

---

## 🧪 Example usage — Inference (programmatic)

Use the same transform / `build_model()` to load a saved fold and predict:

```python
# example (based on label.py/train_model.py)
from PIL import Image
import torch
from label import build_model, transform  # or import equivalent functions from label.py

model = build_model()
model.load_state_dict(torch.load("Folds/seed_XXXX_fold1.pt", map_location="cpu"))
model.eval()

img = Image.open("Model/Images/example.jpg").convert("RGB")
t = transform(img).unsqueeze(0)  # same transform used in label.py
with torch.no_grad():
    score = model(t).item()
print(f"Predicted score: {score:.2f}/5")
```

(Adapt paths & device handling for CUDA as needed.) &#x20;

---

## 🛠 Development & contributions

* Contributions are welcome (bug fixes, more robust preprocessing, fairness audits, alternate architectures).
* Please open issues for feature requests or problems, and submit PRs with tests where possible.
* Suggested improvements:

  * Add dataset balancing & per-group metrics
  * Add unit tests for preprocessing
  * Provide a CLI for inference & a lean, headless evaluation script
  * Add model calibration (e.g., isotonic regression) if you need reliable probabilistic interpretation

---

## 📜 License

This project uses the MIT license (see `LICENSE`).

---

## Acknowledgements

* MediaPipe for face landmark detection used in preprocessing.&#x20;
* EfficientNet (torchvision) backbone for model architecture.&#x20;
