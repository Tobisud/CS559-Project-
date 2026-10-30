
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from model import create_cnn

# 1. Select GPU or CPU
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

# 2. Load trained model
checkpoint = torch.load(
    "gesture_cnn.pth",
    map_location=device,
    weights_only=True
)

class_names = checkpoint["classes"]

model = create_cnn(num_classes=len(class_names))
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

# 3. Same preprocessing as validation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 4. Load test dataset
test_dataset = datasets.ImageFolder(
    root="dataset/test",
    transform=transform
)

assert test_dataset.classes == class_names, \
    "Test dataset classes do not match the trained model."

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

# 5. Predict test images
all_predictions = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        all_predictions.extend(
            predictions.cpu().tolist()
        )
        all_labels.extend(labels.tolist())

# 6. Calculate accuracy
accuracy = accuracy_score(
    all_labels,
    all_predictions
)

print(f"\nTest Accuracy: {accuracy:.2%}")

# 7. Classification report
print("\nClassification Report:")
print(
    classification_report(
        all_labels,
        all_predictions,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0
    )
)

# 8. Confusion matrix
cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(len(class_names)))
)

fig, ax = plt.subplots(figsize=(18, 16))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

display.plot(
    ax=ax,
    cmap="Blues",
    xticks_rotation=90,
    values_format="d",
    colorbar=False
)

plt.title("CNN Gesture Recognition - Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()
