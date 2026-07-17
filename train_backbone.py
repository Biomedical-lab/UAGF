"""Backbone fine-tuning and feature extraction.

Fine-tunes ConvNeXt-V2 Base, Swin-Tiny, or EfficientNet-B3 on an ISIC
dataset and extracts penultimate-layer features for downstream fusion.

Usage:
    python train_backbone.py --config config/isic2018.yaml --backbone convnext --data_dir /path/to/images
    python train_backbone.py --config config/isic2018.yaml --backbone swin --data_dir /path/to/images
    python train_backbone.py --config config/isic2018.yaml --backbone efficientnet --data_dir /path/to/images
"""

import argparse
import os

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Backbone Feature Extraction")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--backbone", type=str, required=True,
                        choices=["convnext", "swin", "efficientnet"])
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to image directory")
    parser.add_argument("--csv_dir", type=str, default=None,
                        help="Path to train/val/test CSV split files")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


BACKBONE_CONFIGS = {
    "convnext": {
        "model_name": "convnextv2_base.fcmae_ft_in22k_in1k",
        "feature_dim": 1024,
        "prefix": "conv_feat_",
        "img_size": 224,
    },
    "swin": {
        "model_name": "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k",
        "feature_dim": 768,
        "prefix": "swin_feat_",
        "img_size": 224,
    },
    "efficientnet": {
        "model_name": "efficientnet_b3.ra2_in1k",
        "feature_dim": 1536,
        "prefix": "effb3_feat_",
        "img_size": 300,
    },
}


class SkinDataset(Dataset):
    """Image dataset for skin lesion classification."""

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def get_transforms(img_size, is_train=True):
    """Get data augmentation transforms."""
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def load_split_csv(csv_path, data_dir):
    """Load image paths and labels from a split CSV file.

    Expects CSV with at least 'image' and 'label' columns.
    """
    df = pd.read_csv(csv_path)
    paths = [os.path.join(data_dir, f"{img}.jpg") for img in df["image"]]
    labels = df["label"].to_numpy().astype(np.int64)
    return paths, labels


def fine_tune(model, train_loader, val_loader, args, device):
    """Fine-tune a backbone model with early stopping.

    Returns:
        Best validation accuracy achieved during training.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_val_loss = float("inf")
    best_acc = 0.0
    wait = 0

    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total * 100
        scheduler.step(avg_val_loss)

        if (epoch + 1) % 5 == 0:
            train_acc = correct / total * 100
            print(
                f"Epoch {epoch+1:3d}/{args.epochs} | "
                f"Train Loss: {train_loss/len(train_loader):.4f} Acc: {train_acc:.1f}% | "
                f"Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.1f}%"
            )

        if avg_val_loss < best_val_loss - 1e-4:
            best_val_loss = avg_val_loss
            best_acc = val_acc
            wait = 0
            best_wts = model.state_dict()
        else:
            wait += 1
            if wait >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    if best_wts is not None:
        model.load_state_dict(best_wts)
    print(f"Best validation accuracy: {best_acc:.2f}%")
    return best_acc


def extract_features(model, dataloader, device, feature_dim, prefix):
    """Extract features from the penultimate layer of a fine-tuned model.

    Uses timm's ``forward_features`` to obtain pre-classifier embeddings,
    then applies global average pooling when spatial dimensions remain.
    """
    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            features = model.forward_features(images)
            if features.dim() > 2:
                features = features.mean(dim=[2, 3]) if features.dim() == 4 else features.mean(dim=1)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    cols = [f"{prefix}{i}" for i in range(feature_dim)]
    df = pd.DataFrame(features[:, :feature_dim], columns=cols)
    df["label"] = labels
    return df


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    backbone_cfg = BACKBONE_CONFIGS[args.backbone]
    num_classes = cfg["num_classes"]
    img_size = backbone_cfg["img_size"]

    print(f"Backbone: {args.backbone} ({backbone_cfg['model_name']})")
    print(f"Device: {device}")
    print(f"Dataset: {cfg['dataset']} ({num_classes} classes)")

    # Create model
    model = timm.create_model(
        backbone_cfg["model_name"],
        pretrained=True,
        num_classes=num_classes,
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Prepare data loaders
    csv_dir = args.csv_dir or args.data_dir
    train_transform = get_transforms(img_size, is_train=True)
    val_transform = get_transforms(img_size, is_train=False)

    splits = {}
    for split_name in ["train", "val", "test"]:
        csv_path = os.path.join(csv_dir, f"{split_name}.csv")
        if os.path.exists(csv_path):
            paths, labels = load_split_csv(csv_path, args.data_dir)
            tfm = train_transform if split_name == "train" else val_transform
            ds = SkinDataset(paths, labels, transform=tfm)
            shuffle = (split_name == "train")
            splits[split_name] = DataLoader(
                ds, batch_size=args.batch_size, shuffle=shuffle,
                num_workers=4, pin_memory=True,
            )
        else:
            print(f"WARNING: {csv_path} not found, skipping {split_name} split.")

    if "train" not in splits or "val" not in splits:
        print("ERROR: train.csv and val.csv are required for fine-tuning.")
        return

    # Fine-tune
    print(f"\nFine-tuning {args.backbone}...")
    fine_tune(model, splits["train"], splits["val"], args, device)

    # Extract features
    print(f"\nExtracting features (dim={backbone_cfg['feature_dim']})...")
    os.makedirs(cfg["feature_dir"], exist_ok=True)
    ffiles = cfg["feature_files"][args.backbone]

    for split_name, loader in splits.items():
        if split_name == "train":
            # Re-create train loader without augmentation for feature extraction
            paths, labels = load_split_csv(
                os.path.join(csv_dir, "train.csv"), args.data_dir,
            )
            ds = SkinDataset(paths, labels, transform=val_transform)
            loader = DataLoader(
                ds, batch_size=args.batch_size, shuffle=False,
                num_workers=4, pin_memory=True,
            )

        df = extract_features(
            model, loader, device,
            backbone_cfg["feature_dim"], backbone_cfg["prefix"],
        )
        out_path = os.path.join(cfg["feature_dir"], ffiles[split_name])
        df.to_csv(out_path, index=False)
        print(f"  {split_name}: {df.shape} -> {out_path}")

    print("\nFeature extraction complete.")


if __name__ == "__main__":
    main()
