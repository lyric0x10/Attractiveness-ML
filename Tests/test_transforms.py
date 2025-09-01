import torchvision.transforms as transforms
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import torch

# === Load a sample image ===
img_path = "Model\\Images\\0C233JspXDJq.jpg"  # Update as needed
img = Image.open(img_path).convert("RGB")

# === Custom transform: Resize with black padding to 224x224 ===
class ResizeWithPadding:
    def __init__(self, size=224, fill_color=(0, 0, 0)):
        self.size = size
        self.fill_color = fill_color

    def __call__(self, img):
        w, h = img.size
        scale = self.size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = img.resize((new_w, new_h), Image.BILINEAR)

        new_img = Image.new("RGB", (self.size, self.size), self.fill_color)
        paste_x = (self.size - new_w) // 2
        paste_y = (self.size - new_h) // 2
        new_img.paste(img_resized, (paste_x, paste_y))
        return new_img

# === Define transforms ===
train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(0.05, 0.05, 0.05, 0.02),  # Less intense jitter
            transforms.RandomPerspective(distortion_scale=0.075, p=1.0),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
            transforms.RandomResizedCrop(224, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
        ])




post_transform = transforms.Compose([
    ResizeWithPadding(size=224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === Apply transforms ===
augmented_img = train_transform(img)
transformed_tensor = post_transform(augmented_img)

# === Function to unnormalize a tensor for display ===
def unnormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

# === Unnormalize and convert for visualization ===
unnorm_img = unnormalize(transformed_tensor.clone(), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
img_np = unnorm_img.permute(1, 2, 0).numpy()
img_np = img_np.clip(0, 1)

# === Display original and transformed images ===
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(img)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Transformed Image")
plt.imshow(img_np)
plt.axis("off")

plt.show()
