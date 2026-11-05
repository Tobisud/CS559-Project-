"""Train an MLP on the hand-landmark CSV made by extract_landmarks.py.

Requires: torch, numpy, pandas, matplotlib
Examples:
    python train_mediapipe.py                                   # hold out people P9, P10 for test, P8 for val
    python train_mediapipe.py --split-mode csv                  # use the train/val/test column from the CSV
    python train_mediapipe.py --test-persons P1 --val-persons P2
    python train_mediapipe.py --use-z                           # also use MediaPipe's (noisy) depth values

Saves:
    gesture_mlp.pth                 checkpoint (weights + classes + settings needed for inference)
    mlp_training_curves.png         loss / accuracy per epoch
    mlp_confusion_matrix.png        test-set confusion matrix
"""

import argparse
import copy
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model_mlp import create_model

NUM_LANDMARKS = 21


# ---------------------------------------------------------------- data
def load_data(csv_path, use_z):
    df = pd.read_csv(csv_path, dtype={"label": str, "person": str, "split": str}, keep_default_na=False)
    axes = ("x", "y", "z") if use_z else ("x", "y")
    cols = [f"n{a}{i}" for i in range(NUM_LANDMARKS) for a in axes]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"CSV is missing normalized columns (e.g. {missing[0]}). Re-run extract_landmarks.py.")
    X = df[cols].to_numpy(np.float32)
    return df, X, len(axes)


def make_splits(df, mode, test_persons, val_persons, seed):
    if mode == "csv":
        split = df["split"].str.lower().replace({"valid": "val", "validation": "val"})
        if not {"train", "val", "test"} <= set(split):
            raise SystemExit("CSV 'split' column needs train/val/test values; use --split-mode person instead.")
        return (split == "train").to_numpy(), (split == "val").to_numpy(), (split == "test").to_numpy()

    if mode == "person":
        persons = df["person"]
        if (persons == "").all():
            raise SystemExit("No person IDs in the CSV; use --split-mode csv or random.")
        test = persons.isin(test_persons).to_numpy()
        val = persons.isin(val_persons).to_numpy()
        if test.sum() == 0 or val.sum() == 0:
            raise SystemExit(f"No rows for test persons {test_persons} or val persons {val_persons}. "
                             f"Available: {sorted(persons.unique())}")
        return ~(test | val), val, test

    # random, stratified by label
    rng = np.random.default_rng(seed)
    train = np.zeros(len(df), bool); val = train.copy(); test = train.copy()
    for _, idx in df.groupby("label").indices.items():
        idx = rng.permutation(idx)
        n_test, n_val = int(len(idx) * 0.15), int(len(idx) * 0.15)
        test[idx[:n_test]] = True
        val[idx[n_test:n_test + n_val]] = True
        train[idx[n_test + n_val:]] = True
    return train, val, test


def augment(x, n_axes, rot_deg, scale_range, noise):
    """Random in-plane rotation (around the wrist), scaling, and jitter on a batch of normalized landmarks."""
    b = x.shape[0]
    pts = x.view(b, NUM_LANDMARKS, n_axes)
    angle = (torch.rand(b, device=x.device) * 2 - 1) * math.radians(rot_deg)
    c, s = torch.cos(angle)[:, None], torch.sin(angle)[:, None]
    px, py = pts[..., 0], pts[..., 1]
    rx, ry = c * px - s * py, s * px + c * py
    scale = torch.empty(b, 1, device=x.device).uniform_(*scale_range)
    out = [rx * scale, ry * scale]
    if n_axes == 3:
        out.append(pts[..., 2] * scale)
    pts = torch.stack(out, dim=-1)
    pts = pts + torch.randn_like(pts) * noise
    return pts.view(b, -1)


# ---------------------------------------------------------------- train / eval
def run_epoch(model, loader, criterion, optimizer=None, aug=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for xb, yb in loader:
            if aug is not None:
                xb = aug(xb)
            logits = model(xb)
            loss = criterion(logits, yb)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            count += len(yb)
    return total_loss / count, correct / count


@torch.no_grad()
def predict(model, X):
    model.eval()
    return model(X).argmax(1).cpu().numpy()


def plot_curves(history, path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ep = range(1, len(history["train_loss"]) + 1)
    a1.plot(ep, history["train_loss"], label="train"); a1.plot(ep, history["val_loss"], label="val")
    a1.set_title("Loss"); a1.set_xlabel("epoch"); a1.legend()
    a2.plot(ep, history["train_acc"], label="train"); a2.plot(ep, history["val_acc"], label="val")
    a2.set_title("Accuracy"); a2.set_xlabel("epoch"); a2.legend()
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_confusion(cm, classes, path):
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes); ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Test confusion matrix (row-normalized)")
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j] and (i == j or norm[i, j] >= 0.05):
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if norm[i, j] > 0.5 else "black")
    fig.colorbar(im, fraction=0.046); fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description="Train an MLP on MediaPipe hand landmarks")
    p.add_argument("--csv", type=Path, default=Path("landmarks.csv"))
    p.add_argument("--output", type=Path, default=Path("gesture_mlp.pth"))
    p.add_argument("--split-mode", choices=["person", "csv", "random"], default="person",
                   help="person: hold out whole people (most honest); csv: use the CSV's split column")
    p.add_argument("--test-persons", nargs="+", default=["P9", "P10"])
    p.add_argument("--val-persons", nargs="+", default=["P8"])
    p.add_argument("--use-z", action="store_true",
                   help="Also use z (depth). Off by default: it hurt accuracy on unseen people in testing")
    p.add_argument("--hidden", type=int, nargs="+", default=[256, 128])
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=20, help="Stop after this many epochs without val improvement")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--rot", type=float, default=15.0, help="Max random rotation in degrees")
    p.add_argument("--noise", type=float, default=0.02, help="Gaussian jitter on normalized coordinates")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    df, X, n_axes = load_data(args.csv, use_z=args.use_z)
    classes = sorted(df["label"].unique(), key=lambda c: (not c.isdigit(), c))   # 0-9 then A-Z
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y = df["label"].map(class_to_idx).to_numpy(np.int64)

    tr, va, te = make_splits(df, args.split_mode, args.test_persons, args.val_persons, args.seed)
    print(f"Classes ({len(classes)}): {' '.join(classes)}")
    print(f"Features per sample: {X.shape[1]}  ({'x,y,z' if n_axes == 3 else 'x,y'})")
    print(f"Split mode: {args.split_mode}  ->  train={tr.sum()}  val={va.sum()}  test={te.sum()}")
    if args.split_mode == "person":
        print(f"  val persons: {args.val_persons}   test persons: {args.test_persons}")

    def tensors(mask):
        return torch.tensor(X[mask], device=device), torch.tensor(y[mask], device=device)

    Xtr, ytr = tensors(tr); Xva, yva = tensors(va); Xte, yte = tensors(te)
    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(Xva, yva), batch_size=1024)

    model = create_model(X.shape[1], len(classes), tuple(args.hidden), args.dropout).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=7)
    aug = None if args.no_augment else (
        lambda xb: augment(xb, n_axes, args.rot, (0.9, 1.1), args.noise))

    history = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc")}
    best_acc, best_state, best_epoch, bad_epochs = -1.0, None, 0, 0
    for epoch in range(1, args.epochs + 1):
        tl, ta = run_epoch(model, train_loader, criterion, optimizer, aug)
        vl, vacc = run_epoch(model, val_loader, criterion)
        scheduler.step(vacc)
        for k, v in zip(history, (tl, ta, vl, vacc)):
            history[k].append(v)
        improved = vacc > best_acc
        if improved:
            best_acc, best_state, best_epoch, bad_epochs = vacc, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            bad_epochs += 1
        if epoch == 1 or epoch % 5 == 0 or improved:
            print(f"Epoch {epoch:3d}  train loss {tl:.4f} acc {ta:.4f} | val loss {vl:.4f} acc {vacc:.4f}"
                  f"  lr {optimizer.param_groups[0]['lr']:.1e}{'  *' if improved else ''}")
        if bad_epochs >= args.patience:
            print(f"Early stopping at epoch {epoch} (best val acc {best_acc:.4f} at epoch {best_epoch})")
            break

    model.load_state_dict(best_state)

    # ---- test
    pred = predict(model, Xte)
    true = yte.cpu().numpy()
    test_acc = (pred == true).mean()
    cm = np.zeros((len(classes), len(classes)), int)
    np.add.at(cm, (true, pred), 1)

    print(f"\nBest val acc: {best_acc:.4f} (epoch {best_epoch})")
    print(f"Test acc:     {test_acc:.4f}  on {len(true)} samples")
    print("\nPer-class test accuracy:")
    per_class = cm.diagonal() / np.maximum(cm.sum(1), 1)
    for i, c in enumerate(classes):
        flag = "  <-- weak" if per_class[i] < 0.9 else ""
        print(f"  {c}: {per_class[i]:.3f}  (n={cm[i].sum()}){flag}")
    off = cm.copy(); np.fill_diagonal(off, 0)
    top = np.dstack(np.unravel_index(np.argsort(off.ravel())[::-1], off.shape))[0][:10]
    print("\nMost common mistakes (true -> predicted):")
    for i, j in top:
        if off[i, j]:
            print(f"  {classes[i]} -> {classes[j]}: {off[i, j]}")

    torch.save({
        "model_state_dict": model.state_dict(),
        "classes": classes,
        "num_inputs": X.shape[1],
        "use_z": n_axes == 3,
        "hidden": list(args.hidden),
        "dropout": args.dropout,
        "val_acc": best_acc,
        "test_acc": float(test_acc),
        "split_mode": args.split_mode,
    }, args.output)
    plot_curves(history, "mlp_training_curves.png")
    plot_confusion(cm, classes, "mlp_confusion_matrix.png")
    print(f"\nSaved model to {args.output.resolve()}")
    print("Saved plots: mlp_training_curves.png, mlp_confusion_matrix.png")


if __name__ == "__main__":
    main()
