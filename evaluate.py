import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from preprocessing import get_eval_transform

DATA_DIR = "dataset_split"
MODEL_PATH = os.path.join("models", "best_model.pth")
BATCH_SIZE = 16


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    print("Class-to-index mapping:", class_to_idx)

    test_dataset = datasets.ImageFolder(
        os.path.join(DATA_DIR, "test"), transform=get_eval_transform()
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Rebuild the exact same architecture used in training
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    target_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="binary", pos_label=class_to_idx["real"])
    recall = recall_score(all_labels, all_preds, average="binary", pos_label=class_to_idx["real"])
    f1 = f1_score(all_labels, all_preds, average="binary", pos_label=class_to_idx["real"])
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=target_names)

    print("\n===== TEST SET RESULTS =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision (real): {precision:.4f}")
    print(f"Recall (real)   : {recall:.4f}")
    print(f"F1-score (real) : {f1:.4f}")

    print("\nConfusion matrix (rows = actual, cols = predicted):")
    print(f"           pred:{target_names[0]:<6} pred:{target_names[1]:<6}")
    for i, row in enumerate(cm):
        print(f"actual:{target_names[i]:<6} {row[0]:<12} {row[1]:<12}")

    print("\nClassification report:")
    print(report)


if __name__ == "__main__":
    main()