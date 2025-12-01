from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple
import os

import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.models import EfficientNet_B7_Weights, efficientnet_b7


@dataclass
class HairTypeModelConfig:
    """Minimal configuration needed to instantiate the head of EfficientNet-B7."""

    num_classes: int = 10
    """Number of hair texture categories (1a-c, 2a, … 4c)."""

    dropout_rate: float = 0.5
    """Dropout applied right before the classification layer."""

    pretrained: bool = True
    """Load ImageNet weights for the EfficientNet-B7 backbone."""

    img_size: int = 600
    """Expected input resolution (height == width == img_size)."""


def build_hair_type_model(config: HairTypeModelConfig) -> nn.Module:
    """Return EfficientNet-B7 adapted to `config.num_classes` hair-type labels."""

    weights = (
        EfficientNet_B7_Weights.IMAGENET1K_V1
        if config.pretrained
        else None
    )
    model = efficientnet_b7(weights=weights)

    classifier = model.classifier
    in_features = classifier[1].in_features
    classifier[0] = nn.Dropout(p=config.dropout_rate, inplace=True)
    classifier[1] = nn.Linear(in_features, config.num_classes)
    model.classifier = classifier
    return model


def freeze_backbone(model: nn.Module, freeze: bool = True) -> None:
    """Freeze or unfreeze the EfficientNet feature extractor."""

    for param in model.features.parameters():
        param.requires_grad = not freeze


def create_training_components(
    model: nn.Module,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
) -> Tuple[nn.Module, AdamW]:
    """Standard optimizer + loss pair for fine-tuning EfficientNet-B7."""

    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    return criterion, optimizer


def train_one_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: AdamW,
    device: Optional[torch.device] = None,
) -> Tuple[float, float]:
    """
    Run a single epoch over ``loader``.

    Returns:
        Average loss and accuracy over the batch dataset.
    """

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        optimizer.zero_grad()
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def evaluate(
    *,
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: Optional[torch.device] = None,
) -> Tuple[float, float]:
    """Evaluate the model in inference mode without updating weights."""

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


def predict(
    *,
    model: nn.Module,
    image: Tensor,
    label_map: Sequence[str],
    device: Optional[torch.device] = None,
) -> Tuple[str, Tensor]:
    """
    Return the predicted label and logits for a single hair image tensor.
    """

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    tensor = image.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)

    idx = logits.argmax(dim=1).item()
    return label_map[idx], logits.squeeze(0)


def save_model_for_deployment(
    *,
    model: nn.Module,
    save_path: str,
    num_classes: int,
    class_names: Sequence[str],
    img_size: int = 600,
) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'num_classes': num_classes,
        'class_names': list(class_names),  # Convert to list for JSON serialization
        'img_size': img_size,
        'model_architecture': 'efficientnet_b7'
    }
    
    torch.save(checkpoint, save_path)
    print(f"Saved deployment model → {save_path}")


def load_model_from_checkpoint(
    *,
    checkpoint_path: str,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, dict]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    num_classes = checkpoint.get('num_classes', 10)
    
    # Build model architecture (same as app does)
    weights = EfficientNet_B7_Weights.IMAGENET1K_V1
    model = efficientnet_b7(weights=weights)
    
    # Replace classifier for hair type classification
    classifier = model.classifier
    in_features = classifier[1].in_features
    classifier[0] = nn.Dropout(p=0.5, inplace=True)
    classifier[1] = nn.Linear(in_features, num_classes)
    model.classifier = classifier
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, checkpoint


if __name__ == "__main__":
    config = HairTypeModelConfig()
    model = build_hair_type_model(config)
    criterion, optimizer = create_training_components(model)

    print("Model ready for training:")
    print(model)
    print(f"{criterion=}, {optimizer=}")

