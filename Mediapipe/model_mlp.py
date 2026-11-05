import torch.nn as nn


class LandmarkMLP(nn.Module):
    """Small MLP that classifies 21 normalized hand landmarks."""

    def __init__(self, num_inputs, num_classes, hidden=(256, 128), dropout=0.3):
        super().__init__()
        layers = []
        prev = num_inputs
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def create_model(num_inputs, num_classes, hidden=(256, 128), dropout=0.3):
    return LandmarkMLP(num_inputs, num_classes, hidden, dropout)
