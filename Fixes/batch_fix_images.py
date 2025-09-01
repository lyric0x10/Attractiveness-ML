import pandas as pd
import os
import cv2
import mediapipe as mp
from math import atan2, degrees
import numpy as np
from PIL import Image

# Setup MediaPipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Resize class to fit image into 244x244 with padding
class ResizeWithPadding:
    def __init__(self, size=244, fill_color=(0, 0, 0)):
        self.size = size
        self.fill_color = fill_color

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = self.size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = img.resize((new_w, new_h), Image.LANCZOS)

        new_img = Image.new("RGB", (self.size, self.size), self.fill_color)
        paste_x = (self.size - new_w) // 2
        paste_y = (self.size - new_h) // 2
        new_img.paste(img_resized, (paste_x, paste_y))
        return new_img

# Get facial landmarks
def get_face_landmarks(image_bgr):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    if results.multi_face_landmarks:
        h, w = image_bgr.shape[:2]
        landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in results.multi_face_landmarks[0].landmark]
        return landmarks
    return None

# Calculate head tilt angle
def correct_head_tilt(landmarks):
    left_eye = landmarks[133]
    right_eye = landmarks[362]
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = degrees(atan2(dy, dx))
    eye_center = ((left_eye[0] + right_eye[0]) // 2,
                  (left_eye[1] + right_eye[1]) // 2)
    return angle, eye_center

# Crop head region with padding
def crop_head_region(image, landmarks, padding_ratio=0.1):
    landmarks_np = np.array(landmarks)
    min_x, min_y = np.min(landmarks_np, axis=0)
    max_x, max_y = np.max(landmarks_np, axis=0)
    width = max_x - min_x
    height = max_y - min_y
    padding_x = int(width * padding_ratio)
    padding_y = int(height * padding_ratio)
    crop_x1 = int(max(0, min_x - padding_x))
    crop_y1 = int(max(0, min_y - padding_y))
    crop_x2 = int(min(image.shape[1], max_x + padding_x))
    crop_y2 = int(min(image.shape[0], max_y + padding_y))
    cropped_image = image[crop_y1:crop_y2, crop_x1:crop_x2]
    adjusted_landmarks = [(x - crop_x1, y - crop_y1) for (x, y) in landmarks]
    return cropped_image, adjusted_landmarks

# Main image processing logic
def process_image_for_display(image_bgr):
    landmarks = get_face_landmarks(image_bgr)
    if landmarks is None:
        return None, None

    angle, center = correct_head_tilt(landmarks)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image_bgr, M, (image_bgr.shape[1], image_bgr.shape[0]))
    rotated_landmarks = [(M[0,0]*x + M[0,1]*y + M[0,2], M[1,0]*x + M[1,1]*y + M[1,2]) for x,y in landmarks]
    cropped, _ = crop_head_region(rotated, rotated_landmarks)
    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(cropped_rgb)
    resized = ResizeWithPadding(size=244, fill_color=(0, 0, 0))(pil_img)
    return resized, cropped

# Process a single image and overwrite if needed
def fix_image_if_needed(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load: {image_path}")
        return

    h, w = image.shape[:2]
    if (h, w) != (244, 244):
        print(f"Processing {os.path.basename(image_path)} - current size: {w}x{h}")
        pil_img, _ = process_image_for_display(image)
        if pil_img is not None:
            cv2.imwrite(image_path, cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR))
        else:
            print(f"  >> Failed to process {os.path.basename(image_path)}")
    else:
        print(f"Skipping {os.path.basename(image_path)} - already 244x244")
        
df = pd.read_csv('Model/Scores.csv', header=None, names=['Image', 'Score'])
image_names = df['Image'].tolist()
# Loop through images in directory
def batch_fix_images():
    for filename in os.listdir(IMG_DIR):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(IMG_DIR, filename)
            if (filename in image_names) == False:
                os.remove(image_path)
            #fix_image_if_needed(image_path)

# Constants
IMG_DIR = "Model/Images"

# Run
if __name__ == "__main__":
    batch_fix_images()
