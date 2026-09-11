from torchvision import transforms

IMAGE_SIZE = 224

# ImageNet mean/std — required because we will use an ImageNet-pretrained
# ResNet18 backbone. The pretrained weights expect inputs normalized with
# these exact statistics; using anything else would hurt transfer learning.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_eval_transform():
    """
    Deterministic pipeline: resize -> tensor -> normalize.
    Used for validation, test, and prediction (no augmentation).
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_train_transform():
    """
    Training pipeline with mild augmentation.

    Deliberately avoids: Gaussian blur, JPEG re-compression, heavy color/hue
    jitter, grayscale conversion, elastic/perspective distortion, and cutout.
    Those can blur or erase the subtle high-frequency artifacts (GAN
    upsampling fingerprints, compression noise patterns) that an AI-vs-real
    detector relies on -- augmenting them away would teach the model to
    ignore the signal that matters most.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.9, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])