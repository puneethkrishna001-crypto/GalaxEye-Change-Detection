import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from dataset import ChangeDetectionDataset
from model import UNetModel

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# Dataset
train_dataset = ChangeDetectionDataset("data/train")

train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True
)

# Model
model = UNetModel().to(device)

# Loss
criterion = nn.BCEWithLogitsLoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
epochs = 2

for epoch in range(epochs):

    model.train()

    running_loss = 0.0

    for images, masks in train_loader:

        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, masks)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)

    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

# Save model
torch.save(model.state_dict(), "outputs/unet_model.pth")

print("Training completed!")