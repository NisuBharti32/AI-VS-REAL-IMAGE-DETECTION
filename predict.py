import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from PIL import Image

from preprocessing import get_eval_transform

MODEL_PATH = os.path.join("models", "best_model.pth")


def load_model(device):
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, idx_to_class


def predict(image_path, model, idx_to_class, device):
    transform = get_eval_transform()  # exact same preprocessing as training/eval

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)  # add batch dimension

    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()

    predicted_class = idx_to_class[pred_idx]
    confidence = probs[pred_idx].item()

    return predicted_class, confidence, {idx_to_class[i]: probs[i].item() for i in range(len(idx_to_class))}


def main():
    if len(sys.argv) != 2:
        print("Usage: python predict.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.isfile(image_path):
        print(f"Error: file not found: {image_path}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, idx_to_class = load_model(device)

    predicted_class, confidence, all_probs = predict(image_path, model, idx_to_class, device)

    print(f"\nImage: {image_path}")
    print(f"Prediction: {predicted_class.upper()}")
    print(f"Confidence: {confidence * 100:.2f}%")
    print("All class probabilities:")
    for class_name, prob in all_probs.items():
        print(f"   {class_name}: {prob * 100:.2f}%")


if __name__ == "__main__":
    main()