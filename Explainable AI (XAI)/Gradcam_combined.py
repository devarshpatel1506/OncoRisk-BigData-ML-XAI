# PyTorch SimpleCNN Grad‐CAM Demo
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

# 1) Load a single image (example)
image_path = "/content/P_00032_1-103.jpg"  # Update path if needed

# 2) Preprocessing transform
def get_transform():
    return transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

# 3) Define a lightweight CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool  = nn.AdaptiveAvgPool2d((1, 1))
        self.fc    = nn.Linear(16, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        self.activations = F.relu(self.conv2(x))
        self.activations.retain_grad()  # Retain grad for Grad-CAM
        x = self.pool(self.activations)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# 4) Load and preprocess the image
transform     = get_transform()
image_pil     = Image.open(image_path).convert("L")
input_tensor  = transform(image_pil).unsqueeze(0)
input_tensor.requires_grad = True

# 5) Initialize model & run a forward + backward pass
model_pt = SimpleCNN()
model_pt.eval()

output = model_pt(input_tensor)
class_idx = output.argmax().item()
model_pt.zero_grad()
output[0, class_idx].backward()

# 6) Build the CAM
grads       = model_pt.activations.grad      # [1, C, H, W]
activations = model_pt.activations.detach()  # [1, C, H, W]
weights     = grads.mean(dim=(2, 3))[0]      # [C]
cam         = torch.zeros(activations.shape[2:], dtype=torch.float32)
for i, w in enumerate(weights):
    cam += w * activations[0, i, :, :]
cam = F.relu(cam)
cam -= cam.min()
cam /= cam.max()
cam_np = cam.numpy()

# 7) Overlay & save
img_cv        = np.array(image_pil.resize((128, 128)))
heatmap       = cv2.resize(cam_np, (128, 128))
heatmap       = np.uint8(255 * heatmap)
heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
overlay       = heatmap_color * 0.4 + cv2.cvtColor(img_cv, cv2.COLOR_GRAY2BGR) * 0.6
overlay       = np.uint8(overlay)
cv2.imwrite("gradcam_output.jpg", overlay)
print("Saved PyTorch Grad-CAM overlay to gradcam_output.jpg")

# TensorFlow H5 Model Grad-CAM Pipeline
# 1) Mount your Drive (Colab)
from google.colab import drive
drive.mount('/content/drive')

# 2) Install dependencies
!pip install --upgrade pip
!pip install tensorflow pillow matplotlib opencv-python

# 3) Imports & paths
import os, glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2

# 4) Define data locations
BASE             = '/content/drive/MyDrive/BigData'
BENIGN_DIR       = os.path.join(BASE, 'benign')
MALIGNANT_DIR    = os.path.join(BASE, 'malignant')
MODEL_PATH       = os.path.join(BASE, 'breast_cancer_model.h5')
OUTPUT_BENIGN    = os.path.join(BASE, 'gradcam_benign')
OUTPUT_MALIGNANT = os.path.join(BASE, 'gradcam_malignant')
os.makedirs(OUTPUT_BENIGN, exist_ok=True)
os.makedirs(OUTPUT_MALIGNANT, exist_ok=True)

# 5) Load & “build” the model
model = load_model(MODEL_PATH)
_     = model.predict(np.zeros((1,224,224,3), dtype=np.float32))  # one dummy pass

# 6) Auto-detect the last Conv2D layer
conv_layers = [l for l in model.layers if isinstance(l, Conv2D)]
if not conv_layers:
    raise RuntimeError("No Conv2D layers found!")
LAST_CONV_LAYER = conv_layers[-1].name
print(" Grad-CAM hook layer:", LAST_CONV_LAYER)

# 7) Image preprocessing
def get_img_array(path, size=(224,224)):
    img = image.load_img(path, target_size=size)
    arr = image.img_to_array(img)
    return np.expand_dims(arr, axis=0)

# 8) Single Grad-CAM function
def make_gradcam_heatmap(img_array):
    grad_model = tf.keras.models.Model(
        inputs  = model.input,
        outputs = [model.get_layer(LAST_CONV_LAYER).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        class_idx = tf.argmax(preds[0])
        loss      = preds[:, class_idx]
    grads = tape.gradient(loss, conv_out)
    pooled= tf.reduce_mean(grads, axis=(0,1,2))
    conv  = conv_out[0]
    cam   = tf.reduce_sum(conv * pooled[..., tf.newaxis], axis=-1)
    cam   = tf.nn.relu(cam) / (tf.reduce_max(cam) + 1e-8)
    return cam.numpy()

# 9) Overlay & save function
def overlay_and_save(src_path, heatmap, dst_path, alpha=0.4):
    orig = cv2.imread(src_path)
    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    hm   = np.uint8(255 * heatmap)
    hm   = cv2.resize(hm, (orig.shape[1], orig.shape[0]))
    hm   = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    hm   = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
    blend= cv2.addWeighted(hm, alpha, orig, 1-alpha, 0)
    cv2.imwrite(dst_path, cv2.cvtColor(blend, cv2.COLOR_RGB2BGR))

# 10) Run over both folders
for src_dir, out_dir in [(BENIGN_DIR, OUTPUT_BENIGN), (MALIGNANT_DIR, OUTPUT_MALIGNANT)]:
    print(f"→ Processing {os.path.basename(src_dir)} …")
    for fn in os.listdir(src_dir):
        src = os.path.join(src_dir, fn)
        dst = os.path.join(out_dir, 'cam_' + fn)
        arr= get_img_array(src)
        hm = make_gradcam_heatmap(arr)
        overlay_and_save(src, hm, dst)
    print(f" Done {os.path.basename(src_dir)}")

print(" All Grad-CAMs saved to:")
print("   ", OUTPUT_BENIGN)
print("   ", OUTPUT_MALIGNANT)


# Mosaic Display of Outputs
import glob
import matplotlib.pyplot as plt
import cv2

# Gather up to 6 PyTorch & TF Grad-CAM outputs
pts = glob.glob('/content/drive/MyDrive/BigData/gradcam_benign/*.jpg')[:3] \
    + glob.glob('/content/drive/MyDrive/BigData/gradcam_malignant/*.jpg')[:3]
if not pts:
    pts = glob.glob('gradcam_output*.jpg')[:6]  # fallback to local

# Plot in 2×3 grid
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for idx, path in enumerate(pts):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    r, c = divmod(idx, 3)
    axes[r, c].imshow(img)
    axes[r, c].set_title(os.path.basename(path))
    axes[r, c].axis('off')
# Hide any extra axes
for idx in range(len(pts), 6):
    r, c = divmod(idx, 3)
    axes[r, c].axis('off')

plt.tight_layout()
plt.show()
