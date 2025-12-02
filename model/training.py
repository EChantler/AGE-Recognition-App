from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.nn as nn
import torch
import os

from face_binary_net import MobilenetBinaryNet
from simple_face_dataset import SimpleFaceDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "data")

import random
import torchvision.transforms as T

# Build file list deterministically
tmp_dataset = SimpleFaceDataset(data_path)
all_samples = tmp_dataset.samples
random.seed(42)
random.shuffle(all_samples)

val_ratio = 0.2
val_size = int(len(all_samples) * val_ratio)
train_samples = all_samples[val_size:]
val_samples = all_samples[:val_size]

# Define transforms: strong augmentation for train, simple resize for val
train_transform = T.Compose([
    T.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = SimpleFaceDataset(samples=train_samples, transform=train_transform)
val_dataset = SimpleFaceDataset(samples=val_samples, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model = MobilenetBinaryNet(pretrained=True).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

best_val_acc = 0.0
for epoch in range(3):  # train longer for better generalization
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    # Train metrics
    train_loss = running_loss/total
    train_acc = correct/total

    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    val_loss_sum = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            val_loss_sum += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
    val_loss = val_loss_sum / val_total if val_total > 0 else 0.0
    val_acc = val_correct / val_total if val_total > 0 else 0.0

    print(
        f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} | "
        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f}"
    )

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        save_path = "./model/face_binary.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Saved best model to: {save_path} (Val Acc: {val_acc:.3f})")

# Debug: visualize one training sample after transform
try:
    batch_images, batch_labels = next(iter(train_loader))
    # Save normalized tensor
    from torchvision.utils import save_image
    import torch
    debug_dir = script_dir
    save_image(batch_images[0], os.path.join(debug_dir, "debug_tensor.png"))
    # Denormalize for viewing
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    denorm = batch_images[0] * std + mean
    save_image(denorm, os.path.join(debug_dir, "debug_tensor_denorm.png"))
    print("Saved debug images: debug_tensor.png, debug_tensor_denorm.png")
except Exception as e:
    print("Debug visualization failed:", e)

