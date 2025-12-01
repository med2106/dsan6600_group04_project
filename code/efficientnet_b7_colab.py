"""
EfficientNet-B7 Training Script for Google Colab
"""

import os
import time
import torch
from torch import nn
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms, datasets
from torchvision.models import efficientnet_b7, EfficientNet_B7_Weights

DATA_DIR = "data/train_val_split/" 

IMG_SIZE = 600
INITIAL_BATCH_SIZE = 4
ACCUMULATION_STEPS = 4
EPOCHS = 20
LR = 3e-4
CHECKPOINT_DIR = "data/checkpoints/efficientnet_b7/"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

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

train_path = os.path.join(DATA_DIR, "train")
val_path = os.path.join(DATA_DIR, "val")

if not os.path.exists(train_path):
    print(f"\nERROR: Training data not found at: {train_path}")
    raise FileNotFoundError(f"Training data directory not found: {train_path}")

if not os.path.exists(val_path):
    print(f"\nERROR: Validation data not found at: {val_path}")
    raise FileNotFoundError(f"Validation data directory not found: {val_path}")

print(f"\nFound training data at: {train_path}")
print(f"Found validation data at: {val_path}")

train_ds = datasets.ImageFolder(train_path, train_tfms)
val_ds = datasets.ImageFolder(val_path, val_tfms)
num_classes = len(train_ds.classes)
print(f"\n✓ Detected {num_classes} classes: {train_ds.classes}")
print(f"  Training samples: {len(train_ds)}")
print(f"  Validation samples: {len(val_ds)}")

def find_largest_batch_size(initial_bs=INITIAL_BATCH_SIZE):
    """Automatically find the largest batch size that fits in GPU memory."""
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
            # Try a forward pass with the model
            with autocast():
                _ = model(images)
            print(f"✓ Batch size {bs} fits in memory.")
            return bs
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"✗ Batch size {bs} OOM → trying smaller batch...")
                torch.cuda.empty_cache()
                bs //= 2
            else:
                raise e
    raise RuntimeError("Could not find any valid batch size.")


weights = EfficientNet_B7_Weights.IMAGENET1K_V1
model = efficientnet_b7(weights=weights)

# Replace classifier for hair type classification
classifier = model.classifier
in_features = classifier[1].in_features
classifier[0] = nn.Dropout(p=0.5, inplace=True)
classifier[1] = nn.Linear(in_features, num_classes)
model.classifier = classifier

model.to(device)
print(f"Model created with {num_classes} output classes")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

print("\n=== Finding best batch size for your GPU ===")
BATCH_SIZE = find_largest_batch_size()
print(f"Using batch size: {BATCH_SIZE}\n")

# Recreate loaders with final batch size
train_loader = torch.utils.data.DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=2, pin_memory=True)
val_loader = torch.utils.data.DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=2, pin_memory=True)


criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scaler = GradScaler()


def validate():
    """Validate the model on validation set."""
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

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"✔ Saved new best model → {ckpt_path}")

print("\nTraining Done! Best Val Accuracy:", best_val_acc)


# Import save function (or define inline if not available)
def save_model_for_deployment(model, save_path, num_classes, class_names, img_size):
    """Save model in format compatible with Streamlit app."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'num_classes': num_classes,
        'class_names': list(class_names),
        'img_size': img_size,
        'model_architecture': 'efficientnet_b7'
    }
    torch.save(checkpoint, save_path)
    print(f"Saved deployment model → {save_path}")

# Save the model in the exact format the Streamlit app expects
final_model_path = os.path.join(CHECKPOINT_DIR, "efficientnet_b7_hair_classifier.pth")
save_model_for_deployment(
    model=model,
    save_path=final_model_path,
    num_classes=num_classes,
    class_names=train_ds.classes,
    img_size=IMG_SIZE
)

# Also save just the state_dict (smaller file, backup option)
state_dict_path = os.path.join(CHECKPOINT_DIR, "efficientnet_b7_state_dict.pth")
torch.save(model.state_dict(), state_dict_path)
print(f"Saved state dict backup → {state_dict_path}")

print(f"\nModel files saved in: {CHECKPOINT_DIR}")
print("Download 'efficientnet_b7_hair_classifier.pth' for Streamlit app!")

