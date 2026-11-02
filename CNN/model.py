"""CNN backbones for gesture classification.

All models are ImageNet-pretrained torchvision models with the final layer
replaced to output `num_classes`. Every model expects 224x224 RGB input
normalized with the ImageNet mean/std, so the same transforms work for all.

Usage:
    model = create_cnn(36, arch="efficientnet_b0", unfreeze=3)
"""

import torch.nn as nn
from torchvision import models

# name -> (builder, pretrained weights)
ARCHS = {
    "mobilenet_v3_small": (models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights.DEFAULT),
    "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT),
    "efficientnet_b2": (models.efficientnet_b2, models.EfficientNet_B2_Weights.DEFAULT),
    "convnext_tiny": (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.DEFAULT),
}

DEFAULT_ARCH = "mobilenet_v3_small"  # what the original gesture_cnn.pth was trained with


def get_head(model, arch):
    """Return the classifier module (the part that is always trained)."""
    return model.fc if arch == "resnet18" else model.classifier


def backbone_blocks(model, arch):
    """Feature-extractor blocks in order from input to output."""
    if arch == "resnet18":
        return [nn.ModuleList([model.conv1, model.bn1]),
                model.layer1, model.layer2, model.layer3, model.layer4]
    return list(model.features.children())


def _replace_last_linear(model, arch, num_classes):
    if arch == "resnet18":
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return
    # Index of the final Linear layer inside each model's classifier
    idx = {"mobilenet_v3_small": 3, "convnext_tiny": 2}.get(arch, 1)  # efficientnet: 1
    old = model.classifier[idx]
    model.classifier[idx] = nn.Linear(old.in_features, num_classes)


def create_cnn(num_classes, arch=DEFAULT_ARCH, unfreeze=0, pretrained=True):
    """Build a classifier.

    unfreeze: how many of the last backbone blocks to train along with the head.
              0  = only the head trains (the original setup, a "linear probe").
              -1 = train every layer (full fine-tuning).
    Note: "N blocks" covers a different share of each model (the last 3 blocks
    are ~80% of MobileNetV3 but ~99% of ResNet-18), so compare models with
    -1 or 0, and use N > 0 only for per-model ablations.
    """
    if arch not in ARCHS:
        raise ValueError(f"Unknown arch {arch!r}. Choose from: {', '.join(ARCHS)}")

    builder, weights = ARCHS[arch]
    model = builder(weights=weights if pretrained else None)

    for param in model.parameters():
        param.requires_grad = False

    _replace_last_linear(model, arch, num_classes)

    for param in get_head(model, arch).parameters():
        param.requires_grad = True

    if unfreeze != 0:
        blocks = backbone_blocks(model, arch)
        for block in (blocks if unfreeze < 0 else blocks[-unfreeze:]):
            for param in block.parameters():
                param.requires_grad = True

    return model


def freeze_bn_stats(model):
    """Keep BatchNorm layers in frozen blocks in eval mode.

    Call right after model.train(). Without this, frozen layers still update
    their running mean/variance during training, which silently changes the
    "frozen" features.
    """
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d) and not any(
            p.requires_grad for p in module.parameters()
        ):
            module.eval()


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
