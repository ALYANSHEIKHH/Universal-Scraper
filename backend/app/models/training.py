import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from PIL import ImageFile
import logging

ImageFile.LOAD_TRUNCATED_IMAGES = True  # prevent crash on bad images

# === Config (can be replaced with import from utils/config.py) ===
DATA_DIR = "data"
MODEL_PATH = "models/cancer_classifier.pth"
BATCH_SIZE = 4
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4
TRAIN_RATIO = 0.8
DEVICE = torch.device("cpu")

logger = logging.getLogger("ModelTraining")


def train_model(data_dir=DATA_DIR, model_path=MODEL_PATH, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, train_ratio=TRAIN_RATIO, device=DEVICE):
    """
    Train a ResNet18 model on images in data_dir. Saves best model to model_path.
    Returns: best_val_acc
    """
    logger.info(f"🚀 Using device: {device}")

    # === Transforms ===
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.CenterCrop(112),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # === Dataset ===
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    logger.info(f"📁 Found {len(dataset)} images across {len(dataset.classes)} classes: {dataset.classes}")

    # === Split ===
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)

    # === Model (Simple + Pretrained) ===
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))
    model = model.to(device)

    # === Loss + Optimizer ===
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # === Training Loop ===
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        total_loss, correct = 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

        train_acc = correct / len(train_dataset)

        # === Validation ===
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_correct += (outputs.argmax(1) == labels).sum().item()

        val_acc = val_correct / len(val_dataset)
        logger.info(f"📅 Epoch [{epoch+1}/{num_epochs}] | Train Loss: {total_loss:.4f} | Train Acc: {train_acc:.2%} | Val Acc: {val_acc:.2%}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save({'model_state_dict': model.state_dict()}, model_path)
            logger.info(f"✅ Saved best model to {model_path}")

    logger.info("🏁 Training Complete")
    return best_val_acc

# CLI entry point
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_model() 