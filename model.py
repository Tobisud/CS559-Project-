
import torch.nn as nn
from torchvision import models

def create_cnn(num_classes):
    weights = models.MobileNet_V3_Small_Weights.DEFAULT

    model = models.mobilenet_v3_small(weights=weights)

    for param in model.features.parameters():
        param.requires_grad = False

    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features,
        num_classes
    )

    return model
