
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from model import create_cnn

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Load checkpoint
checkpoint = torch.load(
    #load model name
    "gesture_cnn.pth",
    map_location=device,
    weights_only=True
)

classes = checkpoint["classes"]

model = create_cnn(len(classes))
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

# Preprocess image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Change this to your image path
image_path = "dataset/test/1/P10_1_940.jpg"

image = Image.open(image_path).convert("RGB")

input_tensor = transform(image).unsqueeze(0).to(device)

# Predict
with torch.no_grad():
    outputs = model(input_tensor)
    probabilities = torch.softmax(outputs, dim=1)

    confidence, predicted_idx = torch.max(
        probabilities, dim=1
    )

predicted_class = classes[predicted_idx.item()]
confidence = confidence.item()

print("Predicted Gesture:", predicted_class)
print(f"Confidence: {confidence:.2%}")

# Display image
plt.imshow(image)
plt.title(
    f"Prediction: {predicted_class}\n"
    f"Confidence: {confidence:.2%}"
)
plt.axis("off")
plt.show()
