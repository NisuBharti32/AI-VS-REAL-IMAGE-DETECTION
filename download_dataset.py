import os
import io
from datasets import load_dataset, Image as HFImage
from PIL import Image

os.makedirs("dataset/real", exist_ok=True)
os.makedirs("dataset/fake", exist_ok=True)

print("Loading dataset (streaming)...")

ds = load_dataset(
    "ComplexDataLab/OpenFake",
    "core",
    split="train",
    streaming=True
)

# Critical fix: disable HF's automatic image decoding.
# Automatic decoding calls image.getexif(), which crashes on records
# with corrupted/non-standard EXIF blocks (the "not a TIFF file" error).
# Turning decode=False gives us raw bytes instead, which we decode
# ourselves with full control over error handling.
ds = ds.cast_column("image", HFImage(decode=False))

real_count = 0
fake_count = 0
skipped = 0

iterator = iter(ds)

while real_count < 500 or fake_count < 500:
    try:
        item = next(iterator)
    except StopIteration:
        print("\nReached end of dataset stream before reaching 500/500.")
        break

    label = item["label"]

    # Skip classes we've already filled
    if label == "real" and real_count >= 500:
        continue
    if label == "fake" and fake_count >= 500:
        continue
    if label not in ("real", "fake"):
        continue

    img_bytes = item["image"]["bytes"]
    if img_bytes is None:
        skipped += 1
        continue

    try:
        image = Image.open(io.BytesIO(img_bytes))
        image.load()
        image = image.convert("RGB")
    except Exception as e:
        skipped += 1
        print(f"\nSkipping corrupted image ({e})")
        continue

    if label == "real":
        image.save(f"dataset/real/{real_count}.jpg", "JPEG")
        real_count += 1
    else:
        image.save(f"dataset/fake/{fake_count}.jpg", "JPEG")
        fake_count += 1

    print(f"Real: {real_count}/500 | Fake: {fake_count}/500 | Skipped: {skipped}", end="\r")

print("\nDataset saved successfully!")
print("Real images:", real_count)
print("Fake images:", fake_count)
print("Skipped (corrupted) records:", skipped)