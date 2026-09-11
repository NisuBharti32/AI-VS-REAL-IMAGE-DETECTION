from datasets import load_dataset

print("Loading a few samples...")

dataset = load_dataset(
    "ComplexDataLab/OpenFake",
    "core",
    split="validation",
    streaming=True
)

print("Dataset loaded successfully.")
print("Features:", dataset.features)

for i, sample in enumerate(dataset.take(3), start=1):
    print(f"\nSample {i}")
    print("Label:", sample["label"])
    print("Model:", sample["model"])
    print("Type:", sample["type"])
    print("Image type:", type(sample["image"]))