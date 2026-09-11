import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from PIL import Image, UnidentifiedImageError
from django.conf import settings
from django.shortcuts import render

from .preprocessing import get_eval_transform

MODEL_PATH = os.path.join(settings.BASE_DIR, "models", "best_model.pth")

# Loaded once when the Django process starts, not on every request.
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_checkpoint = torch.load(MODEL_PATH, map_location=_device)
_class_to_idx = _checkpoint["class_to_idx"]
_idx_to_class = {v: k for k, v in _class_to_idx.items()}

_model = models.resnet18(weights=None)
_num_features = _model.fc.in_features
_model.fc = nn.Linear(_num_features, 2)
_model.load_state_dict(_checkpoint["model_state_dict"])
_model = _model.to(_device)
_model.eval()

# Human-readable labels for the trained classes
DISPLAY_LABELS = {
    "real": "REAL",
    "fake": "AI-GENERATED",
}


def run_prediction(uploaded_file):
    """
    Runs the trained ResNet18 model on an uploaded image file using the
    exact same preprocessing as training/evaluation.
    Raises UnidentifiedImageError/OSError if the file isn't a valid image.
    """
    transform = get_eval_transform()

    image = Image.open(uploaded_file)
    image.load()  # forces decode now, so corrupt/invalid files raise here
    image = image.convert("RGB")

    tensor = transform(image).unsqueeze(0).to(_device)

    with torch.no_grad():
        outputs = _model(tensor)
        probs = F.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()

    predicted_class = _idx_to_class[pred_idx]
    confidence = probs[pred_idx].item() * 100

    return DISPLAY_LABELS.get(predicted_class, predicted_class), confidence


def upload_image(request):
    context = {}

    if request.method == "POST":
        uploaded_file = request.FILES.get("image")

        if not uploaded_file:
            context["error"] = "No image file was received. Please select a file."
        else:
            context["filename"] = uploaded_file.name
            context["filesize"] = uploaded_file.size

            try:
                predicted_label, confidence = run_prediction(uploaded_file)
                context["success"] = True
                context["prediction"] = predicted_label
                context["confidence"] = f"{confidence:.2f}"
            except (UnidentifiedImageError, OSError):
                context["error"] = "The uploaded file is not a valid image. Please upload a JPG or PNG file."
            except Exception:
                context["error"] = "Something went wrong while processing the image. Please try again."

    return render(request, "detector/upload.html", context)