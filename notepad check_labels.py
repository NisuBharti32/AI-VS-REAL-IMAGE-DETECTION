from datasets import load_dataset
from collections import Counter

print("Loading dataset...")

dataset = load_dataset(
    "ComplexDataLab/OpenFake",
    "core",
    split="validation",
    streaming=True
)

labels = Counter()

for sample in dataset.take(1000):
    labels[sample["label"]] += 1

print("Label distribution:")
print(labels)