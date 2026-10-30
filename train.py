
import torch
import torch.nn as nn
from model import create_cnn
import matplotlib.pyplot as plt
import csv
import copy
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load datasets
train_dataset = datasets.ImageFolder(
    root="dataset/train",
    transform=transform
)

val_dataset = datasets.ImageFolder(
    root="dataset/val",
    transform=transform
)

test_dataset = datasets.ImageFolder(
    root="dataset/test",
    transform=transform
)

# Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

print("Classes:", train_dataset.classes)
print("Number of classes:", len(train_dataset.classes))
print("Training images:", len(train_dataset))


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

num_classes = len(train_dataset.classes)

model = create_cnn(num_classes).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.001
)

print("Using device:", device)

history = {
    "train_loss": [],
    "val_loss": [],
    "train_acc": [],
    "val_acc": []
}

epochs = 20
best_val_accuracy = -1.0
best_state = None

for epoch in range(epochs):

    # ---------- TRAINING ----------
    model.train()

    train_loss_sum = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item() * labels.size(0)
        train_correct += (
            outputs.argmax(1) == labels
        ).sum().item()
        train_total += labels.size(0)

    train_loss = train_loss_sum / train_total
    train_acc = train_correct / train_total

    # ---------- VALIDATION ----------
    model.eval()

    val_loss_sum = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss_sum += loss.item() * labels.size(0)
            val_correct += (
                outputs.argmax(1) == labels
            ).sum().item()
            val_total += labels.size(0)

    val_loss = val_loss_sum / val_total
    val_acc = val_correct / val_total

    # ---------- RECORD HISTORY ----------
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Train Acc: {train_acc:.2%} | "
        f"Val Acc: {val_acc:.2%}"
    )

    # Save best model in memory
    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        best_state = copy.deepcopy(model.state_dict())

# ---------- SAVE MODEL ----------
torch.save({
    "model_state_dict": best_state,
    "classes": train_dataset.classes
}, "gesture_cnn.pth")

print("Best validation accuracy:", best_val_accuracy)


epoch_range = range(1, len(history["train_loss"]) + 1)

# Graph 1: Training and validation loss
plt.figure(figsize=(9, 5))

plt.plot(
    epoch_range,
    history["train_loss"],
    marker="o",
    label="Training Loss"
)

plt.plot(
    epoch_range,
    history["val_loss"],
    marker="o",
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Cross Entropy Loss")
plt.title("MobileNetV3 - Training Loss")
plt.legend()
plt.grid(True)

plt.savefig("training_loss.png", dpi=300)
plt.show()

# Graph 2: Training and validation accuracy
plt.figure(figsize=(9, 5))

plt.plot(
    epoch_range,
    history["train_acc"],
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    epoch_range,
    history["val_acc"],
    marker="o",
    label="Validation Accuracy"
)

plt.xlabel("Epoch") 
plt.ylabel("Accuracy")
plt.title("MobileNetV3 - Training Accuracy")
plt.legend()
plt.grid(True)

plt.savefig("training_accuracy.png", dpi=300)
plt.show()
