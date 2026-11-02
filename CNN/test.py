"""Evaluate a trained checkpoint on a test folder.

Examples:
    python test.py --checkpoint results/efficientnet_b0/model.pth
    python test.py --checkpoint results/efficientnet_b0/model.pth --data webcam_test --name webcam

Writes into the checkpoint's folder (suffix from --name, default "test"):
    confusion_matrix_<name>.png
    classification_report_<name>.txt
    metrics_<name>.json   accuracy, top-5 accuracy, macro F1, ms per image
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import DEFAULT_ARCH, create_cnn


def main():
    p = argparse.ArgumentParser(description="Test a gesture CNN")
    p.add_argument("--checkpoint", type=Path, default=Path("gesture_cnn.pth"))
    p.add_argument("--data", type=Path, default=Path("dataset/test"))
    p.add_argument("--name", default="test", help="Suffix for output files, e.g. webcam")
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    class_names = checkpoint["classes"]
    arch = checkpoint.get("arch", DEFAULT_ARCH)  # old checkpoints have no "arch"

    model = create_cnn(len(class_names), arch=arch, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_dataset = datasets.ImageFolder(args.data, transform=transform)

    # Map folder labels to the model's label indices (allows a test folder
    # that only contains some of the classes, e.g. your own webcam captures)
    missing = set(test_dataset.classes) - set(class_names)
    assert not missing, f"Test folder has classes the model doesn't know: {missing}"
    to_model_idx = {i: class_names.index(c) for i, c in enumerate(test_dataset.classes)}
    test_dataset.target_transform = lambda i: to_model_idx[i]

    loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    all_preds, all_labels, all_top5 = [], [], []
    infer_seconds = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            t0 = time.perf_counter()
            outputs = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            infer_seconds += time.perf_counter() - t0
            top5 = outputs.topk(min(5, len(class_names)), dim=1).indices.cpu()
            all_preds.extend(outputs.argmax(1).cpu().tolist())
            all_top5.extend(top5.tolist())
            all_labels.extend(labels.tolist())

    accuracy = accuracy_score(all_labels, all_preds)
    top5_acc = sum(l in t for l, t in zip(all_labels, all_top5)) / len(all_labels)
    present = sorted(set(all_labels))
    macro_f1 = f1_score(all_labels, all_preds, labels=present, average="macro", zero_division=0)
    ms_per_image = 1000 * infer_seconds / len(all_labels)

    print(f"\nModel: {arch}")
    print(f"Test Accuracy: {accuracy:.2%}")
    print(f"Top-5 Accuracy: {top5_acc:.2%}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Inference: {ms_per_image:.2f} ms/image (batch {args.batch_size}, {device})")

    report = classification_report(all_labels, all_preds, labels=present,
                                   target_names=[class_names[i] for i in present],
                                   zero_division=0)
    print("\nClassification Report:\n" + report)

    out_dir = args.checkpoint.parent
    (out_dir / f"classification_report_{args.name}.txt").write_text(report)
    (out_dir / f"metrics_{args.name}.json").write_text(json.dumps({
        "arch": arch, "data": str(args.data), "num_images": len(all_labels),
        "accuracy": accuracy, "top5_accuracy": top5_acc, "macro_f1": macro_f1,
        "ms_per_image": ms_per_image, "device": str(device),
    }, indent=2))

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(18, 16))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
        ax=ax, cmap="Blues", xticks_rotation=90, values_format="d", colorbar=False)
    plt.title(f"{arch} - Confusion Matrix ({args.name})")
    plt.tight_layout()
    plt.savefig(out_dir / f"confusion_matrix_{args.name}.png", dpi=300)
    plt.close()
    print(f"Saved results to {out_dir}/")


if __name__ == "__main__":
    main()
