
import os
import sys
import json
import time
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from torchvision.models import MobileNet_V3_Small_Weights

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

IMAGE_SIZE = 160
BATCH_SIZE = 32
EPOCHS = 3
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_dataset():
    candidates = []

    if len(sys.argv) > 1:
        candidates.append(Path(sys.argv[1]))

    candidates.extend([
        BASE_DIR / "data" / "images",
        BASE_DIR / "food-101" / "images",
        Path.home() / "Downloads" / "Food101_Images" / "food-101" / "images",
        Path.home() / "Downloads" / "Food101_Images" / "images",
        Path.home() / "Downloads" / "food-101" / "images",
    ])

    for candidate in candidates:
        if not candidate.is_dir():
            continue

        class_dirs = [
            p for p in candidate.iterdir()
            if p.is_dir() and p.name not in {"__MACOSX", ".DS_Store"}
        ]

        if len(class_dirs) >= 2:
            return candidate

    raise FileNotFoundError(
        "Food-101 images folder not found.\n"
        "Pass its exact path after train_model.py, for example:\n"
        'py train_model.py "C:\\path\\to\\food-101\\images"'
    )


def main():
    dataset_path = find_dataset()
    print("Dataset:", dataset_path)
    print("Device:", DEVICE)

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_full = datasets.ImageFolder(
        str(dataset_path), transform=train_transform
    )
    val_full = datasets.ImageFolder(
        str(dataset_path), transform=val_transform
    )

    if len(train_full.classes) < 2:
        raise ValueError("Dataset must contain at least two food classes.")

    if train_full.classes != val_full.classes:
        raise ValueError("Training and validation classes do not match.")

    class_names = train_full.classes
    total = len(train_full)
    indices = list(range(total))
    random.shuffle(indices)

    val_count = max(1, int(total * 0.1))
    val_indices = indices[:val_count]
    train_indices = indices[val_count:]

    # Optional quick test: set FOODVISION_MAX_IMAGES=3000
    # to train on a smaller random subset of images.
    max_images = int(os.environ.get("FOODVISION_MAX_IMAGES", "0"))
    if max_images > 0 and len(train_indices) > max_images:
        train_indices = train_indices[:max_images]

    train_data = Subset(train_full, train_indices)
    val_data = Subset(val_full, val_indices)

    workers = 0  # Reliable on Windows.
    train_loader = DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=workers
    )
    val_loader = DataLoader(
        val_data, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=workers
    )

    print("Food classes:", len(class_names))
    print("Training images:", len(train_data))
    print("Validation images:", len(val_data))
    print("Loading pretrained MobileNetV3 weights...")

    model = models.mobilenet_v3_small(
        weights=MobileNet_V3_Small_Weights.DEFAULT
    )

    input_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(
        input_features, len(class_names)
    )
    model = model.to(DEVICE)

    # Train the classifier first for faster CPU training.
    for param in model.features.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(
        model.classifier.parameters(), lr=0.001
    )
    criterion = nn.CrossEntropyLoss()

    best_accuracy = 0.0
    started = time.time()

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for step, (images, labels) in enumerate(train_loader, 1):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if step % 50 == 0 or step == len(train_loader):
                print(
                    f"Epoch {epoch + 1}/{EPOCHS} | "
                    f"Batch {step}/{len(train_loader)} | "
                    f"Loss {loss.item():.4f}"
                )

        model.eval()
        correct = 0
        count = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(images)
                predictions = outputs.argmax(dim=1)
                correct += (predictions == labels).sum().item()
                count += labels.size(0)

        accuracy = 100.0 * correct / max(count, 1)
        avg_loss = running_loss / max(len(train_loader), 1)

        print(
            f"Epoch {epoch + 1}: "
            f"loss={avg_loss:.4f}, "
            f"validation accuracy={accuracy:.2f}%"
        )

        if accuracy >= best_accuracy:
            best_accuracy = accuracy

            torch.save(
                model.state_dict(),
                MODEL_DIR / "foodvision_model.pth"
            )

            with open(
                MODEL_DIR / "class_names.json",
                "w",
                encoding="utf-8"
            ) as f:
                json.dump(class_names, f, indent=2)

    print(f"Training complete in {(time.time() - started) / 60:.1f} minutes.")
    print(f"Best validation accuracy: {best_accuracy:.2f}%")
    print("Saved model:", MODEL_DIR / "foodvision_model.pth")
    print("Saved classes:", MODEL_DIR / "class_names.json")


if __name__ == "__main__":
    main()
