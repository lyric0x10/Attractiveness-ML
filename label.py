import os
import cv2
import mediapipe as mp
from math import atan2, degrees
import numpy as np
import random
import string
import tempfile
import urllib.request
import time
import customtkinter as ctk
from customtkinter import CTkImage
from PIL import Image
import csv
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models.efficientnet import efficientnet_b1, EfficientNet_B1_Weights

# Appearance
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# Directories
IMG_DIR = "Model/Images"
EVAL_DIR = "Model/ToEvaluate"
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)
if not os.path.exists("Model/Scores.csv"):
    with open("Model/Scores.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "rating"])
CSV_PATH = "Model/Scores.csv"

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Face Mesh
mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=True,
                                         max_num_faces=1,
                                         refine_landmarks=True,
                                         min_detection_confidence=0.5)

# Model
class ResizeWithPadding:
    def __init__(self, size=224, fill_color=(0,0,0)):
        self.size = size
        self.fill = fill_color
    def __call__(self, img):
        w, h = img.size
        scale = self.size / max(w, h)
        nw, nh = int(w*scale), int(h*scale)
        img_r = img.resize((nw, nh), Image.LANCZOS)
        bg = Image.new("RGB", (self.size, self.size), self.fill)
        bg.paste(img_r, ((self.size-nw)//2, (self.size-nh)//2))
        return bg

def build_model():
    weights = EfficientNet_B1_Weights.DEFAULT
    model = efficientnet_b1(weights=weights)
    model.classifier[0] = nn.Dropout(0.3)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    return model.to(device)

transform = transforms.Compose([
    ResizeWithPadding(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def gen_name(n=12):
    return ''.join(random.choices(string.ascii_letters+string.digits, k=n))

def download_face(retries=3, delay=1):
    url = 'https://thispersondoesnotexist.com/'
    for _ in range(retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0'})
            data = urllib.request.urlopen(req, timeout=10).read()
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            tmp.write(data)
            tmp.close()
            return tmp.name
        except:
            time.sleep(delay)
    raise RuntimeError("Download failed")

# Image processing

def get_landmarks(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res = mp_face.process(rgb)
    if not res.multi_face_landmarks:
        return None
    h, w = img_bgr.shape[:2]
    return [(int(lm.x*w), int(lm.y*h)) for lm in res.multi_face_landmarks[0].landmark]

def process_for_display(img_bgr):
    landmarks = get_landmarks(img_bgr)
    if not landmarks:
        return None, None
    # compute tilt
    l_eye, r_eye = landmarks[133], landmarks[362]
    dy, dx = r_eye[1] - l_eye[1], r_eye[0] - l_eye[0]
    ang = degrees(atan2(dy, dx))
    cen = ((l_eye[0] + r_eye[0]) // 2, (l_eye[1] + r_eye[1]) // 2)
    M = cv2.getRotationMatrix2D(cen, ang, 1)
    rot = cv2.warpAffine(img_bgr, M, (img_bgr.shape[1], img_bgr.shape[0]))
    pts = [(M[0,0]*x + M[0,1]*y + M[0,2], M[1,0]*x + M[1,1]*y + M[1,2]) for x, y in landmarks]
    arr = np.array(pts)
    x1_f, y1_f = arr.min(axis=0)
    x2_f, y2_f = arr.max(axis=0)
    # ensure integer indices
    x1, y1, x2, y2 = map(int, [x1_f, y1_f, x2_f, y2_f])
    pad = 0.1
    dx_ = int((x2 - x1) * pad)
    dy_ = int((y2 - y1) * pad)
    y_start = max(0, y1 - dy_)
    y_end = min(rot.shape[0], y2 + dy_)
    x_start = max(0, x1 - dx_)
    x_end = min(rot.shape[1], x2 + dx_)
    crop = rot[y_start:y_end, x_start:x_end]
    if crop.size == 0:
        return None, None
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    return ResizeWithPadding(size=244)(pil), crop

def fix_image_if_needed(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load: {image_path}")
        return

    h, w = image.shape[:2]
    if (h, w) != (244, 244):
        print(f"Processing {os.path.basename(image_path)} - current size: {w}x{h}")
        pil_img, _ = process_for_display(image)
        if pil_img is not None:
            cv2.imwrite(image_path, cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR))
        else:
            print(f"  >> Failed to process {os.path.basename(image_path)}")
    else:
        print(f"Skipping {os.path.basename(image_path)} - already 244x244")

def process_and_save(path):
    img = cv2.imread(path)
    disp, crop = process_for_display(img)
    if crop is None:
        return None
    name = gen_name() + '.jpg'
    cv2.imwrite(os.path.join(IMG_DIR, name), crop)
    fix_image_if_needed(os.path.join(IMG_DIR, name))
    return name

def predict(path, model):
    img = Image.open(path).convert('RGB')
    t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        p = model(t).item()
    return max(0.0, min(5.0, p))

def record(fname, rate):
    with open(CSV_PATH, 'a', newline='') as f:
        csv.writer(f).writerow([fname, rate])

class FaceRater(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Face Rater")
        self.geometry("775x470")
        self.model = build_model()
        self.model.load_state_dict(torch.load("Folds/seed_3516510858_fold1.pt", map_location=device))
        self.model.eval()

        # UI Components
        self.img_label = ctk.CTkLabel(self, text='', width=250, height=250)
        self.img_label.pack(pady=(20,10))
        self.pred_label = ctk.CTkLabel(self, text='', font=('Arial', 16))
        self.pred_label.pack(pady=(0,20))

        # Rating Buttons
        btns = ctk.CTkFrame(self)
        btns.pack(pady=10)
        labels = ["Very Unattractive","Unattractive","Average","Attractive","Very Attractive"]
        for i, text in enumerate(labels, start=1):
            b = ctk.CTkButton(btns, text=text, command=lambda v=i: self.rate(v))
            b.grid(row=0, column=i-1, padx=5)

        # Control Buttons
        ctrl = ctk.CTkFrame(self)
        ctrl.pack(pady=10)
        ctk.CTkButton(ctrl, text="Skip (S)", command=self.next).pack(side='left', padx=10)
        ctk.CTkButton(ctrl, text="Delete (D)", command=self.delete, fg_color="red").pack(side='left')

        # Key Bindings
        self.bind('s', lambda e: self.next())
        self.bind('d', lambda e: self.delete())
        for i in range(1, 6):
            self.bind(str(i), lambda e, v=i: self.rate(v))

        self.queue = self._load_queue()
        self.next()

    def _load_queue(self):
        files = [os.path.join(EVAL_DIR, f) for f in os.listdir(EVAL_DIR)
                 if f.lower().endswith(('.jpg','.png','.jpeg'))]
        random.shuffle(files)
        return files

    def next(self):
        if self.queue:
            path = self.queue.pop()
            self.from_ai = False
        else:
            path = download_face()
            self.from_ai = True
        self.current = path

        img_bgr = cv2.imread(path)
        disp, _ = process_for_display(img_bgr)
        if not disp:
            pil = Image.open(path).convert('RGB')
            pil.thumbnail((244,244))
            disp = ResizeWithPadding(size=244)(pil)
        self.photo = CTkImage(disp, size=(244,244))
        self.img_label.configure(image=self.photo)

        # Processing + Prediction
        if self.from_ai:
            processed_path = process_and_save(path)  # Save to IMG_DIR
        else:
            processed_path = process_and_save(path)

        self.saved_name = processed_path  # Store filename
        score_path = os.path.join(IMG_DIR, processed_path) if processed_path else path
        raw = predict(score_path, self.model)
        self.pred_label.configure(text=f"Score: {raw:.2f}/5")


    def rate(self, rating):
        if self.saved_name:
            record(self.saved_name, rating)
        if self.current and os.path.exists(self.current):
            os.remove(self.current)
        self.next()


    def delete(self):
        # Delete original
        if self.current and os.path.exists(self.current):
            os.remove(self.current)

        # Delete processed IMG_DIR version if it exists
        if self.saved_name:
            saved_path = os.path.join(IMG_DIR, self.saved_name)
            if os.path.exists(saved_path):
                os.remove(saved_path)

        self.next()


if __name__ == '__main__':
    FaceRater().mainloop()
