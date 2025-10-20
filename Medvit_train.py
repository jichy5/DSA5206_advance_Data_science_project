
import os
import sys
import json
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm

# 
from MedViT import MedViT_small, MedViT_base, MedViT_large, MedViT_tiny


def main():
    # ================================
    # Device setup
    # ================================
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ================================
    # Dataset & Transform
    # ================================
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))
        ]),
        "val": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))
        ])
    }

    train_dir = "/workspace/MedMNIST_png/dermamnist_train"
    val_dir = "/workspace/MedMNIST_png/dermamnist_val"

    assert os.path.exists(train_dir), f"{train_dir} not found"
    assert os.path.exists(val_dir), f"{val_dir} not found"

    train_dataset = datasets.ImageFolder(root=train_dir, transform=data_transform["train"])
    val_dataset = datasets.ImageFolder(root=val_dir, transform=data_transform["val"])
    train_num = len(train_dataset)
    val_num = len(val_dataset)
    print(f" Using {train_num} images for training, {val_num} for validation.")

    # ================================
    # Save class indices
    # ================================
    class_dict = {v: k for k, v in train_dataset.class_to_idx.items()}
    with open("class_indices.json", "w") as f:
        json.dump(class_dict, f, indent=4)

    # ================================
    # DataLoader
    # ================================
    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print(f"📦 Using {nw} dataloader workers per process.")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=nw)

    # ================================
    # Model Setup
    # ================================
    num_classes = 7
    model_type = "tiny"  # 可选: "small" / "base" / "large"

    if model_type == "tiny":
        model = MedViT_tiny(num_classes=num_classes).to(device)
    elif model_type == "small":
        model = MedViT_small(num_classes=num_classes).to(device)
    elif model_type == "base":
        model = MedViT_base(num_classes=num_classes).to(device)
    else:
        model = MedViT_large(num_classes=num_classes).to(device)

    print(f"Loaded model: MedViT_{model_type}")


    # ================================
    # Training Configuration
    # ================================
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    epochs = 100
    best_acc = 0.0
    save_path = f"./dermamnist_224_medvit_{model_type}.pth"

    # ================================
    # Training Loop
    # ================================
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)

        for step, (imgs, labels) in enumerate(train_bar):
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_bar.desc = f"Train epoch[{epoch+1}/{epochs}] loss:{loss:.3f}"

        # ================================
        # Validation Phase
        # ================================
        model.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout)
            for imgs, labels in val_bar:
                outputs = model(imgs.to(device))
                preds = torch.max(outputs, dim=1)[1]
                acc += torch.eq(preds, labels.to(device)).sum().item()

        val_acc = acc / val_num
        print(f"[epoch {epoch+1}] train_loss: {running_loss/len(train_loader):.3f}  val_acc: {val_acc:.3f}")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model with acc={best_acc:.4f}")

    print(f"Best validation accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
