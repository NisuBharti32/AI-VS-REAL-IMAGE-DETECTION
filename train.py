import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models

from preprocessing import get_train_transform, get_eval_transform

DATA_DIR = "dataset_split"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")

BATCH_SIZE = 16
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
PATIENCE = 5  # stop if val loss doesn't improve for this many epochs


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = datasets.ImageFolder(
        os.path.join(DATA_DIR, "train"), transform=get_train_transform()
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(DATA_DIR, "val"), transform=get_eval_transform()
    )

    print("Class-to-index mapping:", train_dataset.class_to_idx)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ResNet18 pretrained on ImageNet, final layer swapped for 2-class output
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)

    # Freeze the pretrained backbone except the last residual block (layer4).
    # A fully frozen backbone plateaus around 60-65% here because generic
    # ImageNet features are tuned for semantic content, not the subtle
    # texture/frequency artifacts that separate real vs AI-generated images.
    # Unfreezing layer4 lets the network adapt its higher-level filters to
    # this task, while earlier layers (generic edges/colors/textures) stay
    # frozen to limit overfitting risk on a small dataset.
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    # Smaller LR for layer4 (already holds useful pretrained weights),
    # larger LR for the new randomly-initialized final layer.
    optimizer = optim.Adam([
        {"params": model.layer4.parameters(), "lr": 1e-5},
        {"params": model.fc.parameters(), "lr": LEARNING_RATE},
    ], weight_decay=1e-4)

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        # ---- Train ----
        model.train()
        running_loss, running_correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = running_correct / total

        # ---- Validate ----
        model.eval()
        val_loss_sum, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss_sum += loss.item() * images.size(0)
                val_correct += (outputs.argmax(dim=1) == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total

        print(f"Epoch {epoch}/{NUM_EPOCHS} | "
              f"Train loss: {train_loss:.4f} acc: {train_acc:.4f} | "
              f"Val loss: {val_loss:.4f} acc: {val_acc:.4f}")

        # ---- Early stopping + checkpointing ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "class_to_idx": train_dataset.class_to_idx,
            }, MODEL_PATH)
            print(f"  -> New best model saved (val loss: {val_loss:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"\nEarly stopping triggered after {epoch} epochs "
                      f"(no val improvement for {PATIENCE} epochs).")
                break

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()