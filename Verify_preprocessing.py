import os
from PIL import Image
from preprocessing import get_eval_transform

DATASET_DIR = "dataset"
CLASSES = ["real", "fake"]


def main():
    transform = get_eval_transform()

    total = 0
    failed = []

    for class_name in CLASSES:
        folder = os.path.join(DATASET_DIR, class_name)
        files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        for fname in files:
            path = os.path.join(folder, fname)
            total += 1
            try:
                with Image.open(path) as img:
                    img = img.convert("RGB")
                    tensor = transform(img)
                    # Sanity check on final tensor shape
                    assert tensor.shape == (3, 224, 224)
            except Exception as e:
                failed.append((class_name, fname, str(e)))

    print(f"Checked {total} images.")
    if failed:
        print(f"{len(failed)} images failed preprocessing:")
        for class_name, fname, err in failed:
            print(f"   - {class_name}/{fname}: {err}")
    else:
        print("All images passed: RGB conversion, resize to 224x224, and normalization all succeeded.")
        print("No processed copies were saved — preprocessing will be applied on-the-fly during training.")


if __name__ == "__main__":
    main()