## optimized to run EfficientNet-V2-L on limited GPU memory for Google Colab by finding max batch size

import os
import math
import time
import torch
import shutil
from torch import nn
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms, datasets
from torchvision.models import efficientnet_v2_l, EfficientNet_V2_L_Weights

# ============================================================
# CONFIGURATION
# ============================================================
DATA_DIR = "data/augmented/"     # root folder with train/val subfolders --> 
IMG_SIZE = 600                      # EfficientNet-V2-L default is ~512, 600 is fine
INITIAL_BATCH_SIZE = 8              # script will automatically reduce if needed
ACCUMULATION_STEPS = 4              # effectively batch_size * accumulation
EPOCHS = 20
LR = 3e-4
CHECKPOINT_DIR = "data/checkpoints/efficientnet_v2_l/"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ============================================================
# DATA AUGMENTATION & LOADING
# ============================================================
train_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), train_tfms)
val_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), val_tfms)
num_classes = len(train_ds.classes)
print(f"Detected {num_classes} classes: {train_ds.classes}")

# Loader is created AFTER we determine batch size


# ============================================================
# FUNCTION: FIND MAXIMUM BATCH SIZE AUTOMATICALLY
# ============================================================
def find_largest_batch_size(initial_bs=INITIAL_BATCH_SIZE):
    bs = initial_bs
    while bs > 0:
        try:
            test_loader = torch.utils.data.DataLoader(
                train_ds,
                batch_size=bs,
                shuffle=True,
                num_workers=2,
                pin_memory=True
            )
            images, labels = next(iter(test_loader))
            images = images.to(device)
            labels = labels.to(device)
            # Try a forward pass
            with autocast():
                _ = images.mean() + labels.float().mean()
            print(f"Batch size {bs} fits in memory.")
            return bs
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Batch size {bs} OOM → trying smaller batch...")
                torch.cuda.empty_cache()
                bs //= 2
            else:
                raise e
    raise RuntimeError("Could not find any valid batch size.")


print("\n=== Finding best batch size for your GPU ===")
BATCH_SIZE = find_largest_batch_size()
print(f"✔ Using batch size: {BATCH_SIZE}\n")

# Recreate loaders with final batch size
train_loader = torch.utils.data.DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=2, pin_memory=True)
val_loader = torch.utils.data.DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=2, pin_memory=True)

# ============================================================
# MODEL
# ============================================================
weights = EfficientNet_V2_L_Weights.IMAGENET1K_V1
model = efficientnet_v2_l(weights=weights)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=LR)
scaler = GradScaler()

# ============================================================
# TRAINING LOOP
# ============================================================
def validate():
    model.eval()
    total, correct = 0, 0
    running_loss = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            with autocast():
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)

    return running_loss / len(val_loader), correct / total


best_val_acc = 0.0

print("\n=== Starting Training ===\n")
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0
    start = time.time()

    optimizer.zero_grad()

    for i, (imgs, labels) in enumerate(train_loader):
        imgs, labels = imgs.to(device), labels.to(device)

        with autocast():
            outputs = model(imgs)
            loss = criterion(outputs, labels) / ACCUMULATION_STEPS

        scaler.scale(loss).backward()

        if (i + 1) % ACCUMULATION_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item() * ACCUMULATION_STEPS

    val_loss, val_acc = validate()
    duration = time.time() - start

    print(f"Epoch {epoch}/{EPOCHS} | "
          f"Train Loss: {running_loss/len(train_loader):.4f} | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
          f"Time: {duration:.1f}s")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"✔ Saved new best model → {ckpt_path}")

print("\nTraining Done! Best Val Accuracy:", best_val_acc)
