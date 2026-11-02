"""Train a gesture CNN on dataset/train and pick the best epoch on dataset/val.

Examples:
    python train.py --model efficientnet_b0
    python train.py --model resnet18
    python train.py --model mobilenet_v3_small --unfreeze 0 --no-augment --tag mnv3_original

Every run writes to results/<tag>/ (tag defaults to the model name):
    model.pth            best checkpoint (also stores the model name and classes)
    history.csv          loss/accuracy per epoch, for the report
    training_loss.png    loss curves
    training_accuracy.png
    run_info.txt         settings, parameter counts, training time
"""

import argparse
import copy
import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import ARCHS, count_parameters, create_cnn, freeze_bn_stats, get_head

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args():
    p = argparse.ArgumentParser(description="Train a gesture CNN")
    p.add_argument("--model", default="efficientnet_b0", choices=list(ARCHS))
    p.add_argument("--data", type=Path, default=Path("dataset"),
                   help="Folder containing train/ and val/")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the new head")
    p.add_argument("--backbone-lr", type=float, default=1e-4,
                   help="Learning rate for unfrozen backbone blocks")
    p.add_argument("--unfreeze", type=int, default=-1,
                   help="Last N backbone blocks to train: -1 = all (default), 0 = head only")
    p.add_argument("--no-augment", action="store_true", help="Disable training augmentation")
    p.add_argument("--no-pretrained", action="store_true", help="Train from random weights")
    p.add_argument("--workers", type=int, default=2, help="DataLoader worker processes")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", default=None, help="Run name for the output folder")
    return p.parse_args()


def build_transforms(augment):
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    if not augment:
        return eval_tf, eval_tf
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        # No horizontal flip: mirroring can change the meaning of some signs.
        transforms.RandomAffine(degrees=10, translate=(0.10, 0.10), scale=(0.9, 1.1),
                                fill=(190, 190, 190)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    if training:
        model.train()
        freeze_bn_stats(model)
    else:
        model.eval()

    loss_sum, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return loss_sum / total, correct / total


def save_plot(epochs, train_vals, val_vals, ylabel, title, path):
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, train_vals, marker="o", label=f"Training {ylabel}")
    plt.plot(epochs, val_vals, marker="o", label=f"Validation {ylabel}")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    run_name = args.tag or args.model
    out_dir = Path("results") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    train_tf, eval_tf = build_transforms(augment=not args.no_augment)
    train_dataset = datasets.ImageFolder(args.data / "train", transform=train_tf)
    val_dataset = datasets.ImageFolder(args.data / "val", transform=eval_tf)
    assert train_dataset.classes == val_dataset.classes, "train/ and val/ have different classes"
    classes = train_dataset.classes

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_cnn(len(classes), arch=args.model, unfreeze=args.unfreeze,
                       pretrained=not args.no_pretrained).to(device)
    total_params, trainable_params = count_parameters(model)

    print(f"Model: {args.model} | device: {device}")
    print(f"Classes ({len(classes)}): {classes}")
    print(f"Train images: {len(train_dataset)} | Val images: {len(val_dataset)}")
    print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Head gets a higher learning rate than the pretrained layers
    head_ids = {id(p) for p in get_head(model, args.model).parameters()}
    head_params = [p for p in model.parameters() if p.requires_grad and id(p) in head_ids]
    backbone_params = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    param_groups = [{"params": head_params, "lr": args.lr}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": args.backbone_lr})
    optimizer = torch.optim.Adam(param_groups)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc, best_epoch, best_state = -1.0, 0, None
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)

        for key, value in zip(history, (train_loss, val_loss, train_acc, val_acc)):
            history[key].append(value)

        print(f"Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Train Acc: {train_acc:.2%} | Val Acc: {val_acc:.2%}")

        if val_acc > best_val_acc:
            best_val_acc, best_epoch = val_acc, epoch
            best_state = copy.deepcopy(model.state_dict())

    train_minutes = (time.time() - start) / 60

    torch.save({
        "model_state_dict": best_state,
        "classes": classes,
        "arch": args.model,
    }, out_dir / "model.pth")

    with open(out_dir / "history.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", *history])
        for i in range(args.epochs):
            writer.writerow([i + 1, *(history[k][i] for k in history)])

    epochs = range(1, args.epochs + 1)
    save_plot(epochs, history["train_loss"], history["val_loss"], "Loss",
              f"{args.model} - Training Loss", out_dir / "training_loss.png")
    save_plot(epochs, history["train_acc"], history["val_acc"], "Accuracy",
              f"{args.model} - Training Accuracy", out_dir / "training_accuracy.png")

    with open(out_dir / "run_info.txt", "w") as f:
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
        f.write(f"total_params: {total_params}\ntrainable_params: {trainable_params}\n")
        f.write(f"best_val_acc: {best_val_acc:.4f}\nbest_epoch: {best_epoch}\n")
        f.write(f"train_minutes: {train_minutes:.1f}\n")

    print(f"Best validation accuracy: {best_val_acc:.2%} (epoch {best_epoch})")
    print(f"Saved results to {out_dir}/")


if __name__ == "__main__":
    main()
