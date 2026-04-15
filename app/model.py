import torch
from torchvision import models, transforms
from PIL import Image
import os
import json

# -----------------------
# MODEL
# -----------------------
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()
model.to("cpu")

# -----------------------
# TRANSFORM
# -----------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# -----------------------
# LOAD LABELS (LOCAL FILE)
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
labels_path = os.path.join(BASE_DIR, "..", "labels.json")

with open(labels_path, "r") as f:
    labels = json.load(f)

# -----------------------
# PREDICTION FUNCTION
# -----------------------
def predict_image(image: Image.Image):
    img = image.convert("RGB")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top5_prob, top5_catid = torch.topk(probabilities, 5)

    results = []

    for i in range(5):
        results.append({
            "label": labels[top5_catid[i].item()],
            "confidence": float(top5_prob[i].item())
        })

    return results