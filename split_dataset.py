import os
import random
import shutil

SOURCE_DIR = "dataset"
OUTPUT_DIR = "dataset_split"
CLASSES = ["real", "fake"]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42


def split_list(files, train_ratio, val_ratio):
    n = len(files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    # remainder goes to test, so all files are accounted for
    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]
    return train_files, val_files, test_files


def main():
    random.seed(SEED)

    for split in ["train", "val", "test"]:
        for class_name in CLASSES:
            os.makedirs(os.path.join(OUTPUT_DIR, split, class_name), exist_ok=True)

    summary = {}

    for class_name in CLASSES:
        src_folder = os.path.join(SOURCE_DIR, class_name)
        files = [f for f in os.listdir(src_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        files.sort()          # deterministic order before shuffling
        random.shuffle(files) # seeded shuffle -> reproducible split

        train_files, val_files, test_files = split_list(files, TRAIN_RATIO, VAL_RATIO)

        for split_name, split_files in [("train", train_files), ("val", val_files), ("test", test_files)]:
            dest_folder = os.path.join(OUTPUT_DIR, split_name, class_name)
            for fname in split_files:
                shutil.copy2(
                    os.path.join(src_folder, fname),
                    os.path.join(dest_folder, fname)
                )

        summary[class_name] = {
            "train": len(train_files),
            "val": len(val_files),
            "test": len(test_files),
            "total": len(files),
        }

    print("Split complete. Originals in 'dataset/' were not modified.\n")
    print(f"{'Class':<8}{'Train':<8}{'Val':<8}{'Test':<8}{'Total':<8}")
    for class_name, counts in summary.items():
        print(f"{class_name:<8}{counts['train']:<8}{counts['val']:<8}{counts['test']:<8}{counts['total']:<8}")


if __name__ == "__main__":
    main()