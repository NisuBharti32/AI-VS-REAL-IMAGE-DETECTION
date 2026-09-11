from huggingface_hub import HfApi

DATASET_ID = "ComplexDataLab/OpenFake"

api = HfApi()

info = api.dataset_info(DATASET_ID)

print("Dataset ID:", info.id)
print("Downloads:", getattr(info, "downloads", None))
print("Likes:", getattr(info, "likes", None))

print("\nDataset tags:")
for tag in (getattr(info, "tags", None) or []):
    print("-", tag)

print("\nCard data:")
print(getattr(info, "cardData", None))