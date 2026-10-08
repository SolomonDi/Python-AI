import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

DATA_DIR = "data/chest_xray"
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4

MEAN = [0.5]
STD = [0.5]

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

eval_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def get_dataLoader():
    train_full = datasets.ImageFolder(root=f"{DATA_DIR}/train", transform=train_transform)
    val_full = datasets.ImageFolder(root=f"{DATA_DIR}/train", transform=eval_transform)
    test_dataset = datasets.ImageFolder(root=f"{DATA_DIR}/test", transform=eval_transform)

    indices = list(range(len(train_full)))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.15, stratify=train_full.targets, random_state=42
    )
    train_dataset = Subset(train_full, train_idx)
    val_dataset = Subset(val_full, val_idx)

    loader_kwargs = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                         pin_memory=torch.cuda.is_available(), persistent_workers=NUM_WORKERS > 0)
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, **loader_kwargs)
    test_loader = DataLoader(test_dataset, **loader_kwargs)

    return train_loader, val_loader, test_loader, train_full.classes


if __name__ == "__main__":
    train_loader, val_loader, test_loader, classes = get_dataLoader()
    print("Classes:", classes)
    print("Number of train batches:", len(train_loader))
    print("Number of validation batches:", len(val_loader))
    print("Number of test batches:", len(test_loader))
