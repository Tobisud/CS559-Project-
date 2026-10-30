import copy
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import create_cnn

# ---------- CONFIG ----------
#model name
MODEL_PATH = "gesture_cnn.pth"
#fine tune model name
OUTPUT_PATH = "gesture_cnn_finetuned.pth"

EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.00001

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

# ---------- LOAD ORIGINAL MODEL ----------
checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

classes = checkpoint["classes"]

model = create_cnn(num_classes=len(classes))
model.load_state_dict(checkpoint["model_state_dict"])
# Unfreeze last 3 CNN feature blocks
for layer in list(model.features.children())[-3:]:
    for param in layer.parameters():
        param.requires_grad = True

# Make sure classifier is trainable
for param in model.classifier.parameters():
    param.requires_grad = True
model = model.to(device)

# ---------- TRANSFORMS ----------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    # Random rotation, shifting and scaling
    transforms.RandomAffine(
        degrees=10,
        translate=(0.10, 0.10),
        scale=(0.9, 1.1),
        fill=(190, 190, 190)
    ),

    # Random brightness and contrast
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.1
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------- LOAD DATASETS ----------
train_dataset = datasets.ImageFolder(
    "finetune_dataset/train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    "finetune_dataset/val",
    transform=val_transform
)

# # Check class order
# assert train_dataset.classes == classes
# assert val_dataset.classes == classes
#training for one dataset only
# Original class mapping from the trained model
class_to_idx = {name: i for i, name in enumerate(classes)}

# Verify webcam classes exist in the original model
assert set(train_dataset.classes).issubset(set(classes))
assert set(val_dataset.classes).issubset(set(classes))

# Map webcam dataset labels to original model labels
train_dataset.target_transform = (
    lambda idx: class_to_idx[train_dataset.classes[idx]]
)

val_dataset.target_transform = (
    lambda idx: class_to_idx[val_dataset.classes[idx]]
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ---------- LOSS AND OPTIMIZER ----------
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=LEARNING_RATE
)

# ---------- TRAINING ----------
history = {
    "train_loss": [],
    "val_loss": [],
    "train_acc": [],
    "val_acc": []
}

best_val_acc = -1.0
best_weights = copy.deepcopy(model.state_dict())

for epoch in range(EPOCHS):

    model.train()

    train_loss_sum = 0
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

    val_loss_sum = 0
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

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Train Acc: {train_acc:.2%} | "
        f"Val Acc: {val_acc:.2%}"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_weights = copy.deepcopy(model.state_dict())

# ---------- SAVE ----------
torch.save({
    "model_state_dict": best_weights,
    "classes": classes
}, OUTPUT_PATH)

print(f"Best validation accuracy: {best_val_acc:.2%}")
print(f"Saved: {OUTPUT_PATH}")

# ---------- PLOT ----------
epochs = range(1, EPOCHS + 1)

plt.figure(figsize=(9, 5))
plt.plot(epochs, history["train_loss"], label="Training Loss")
plt.plot(epochs, history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CNN Fine-Tuning Loss")
plt.legend()
plt.grid(True)
plt.savefig("finetuning_loss.png", dpi=300)
plt.show()
