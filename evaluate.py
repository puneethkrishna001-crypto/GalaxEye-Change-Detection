import torch
from torch.utils.data import DataLoader
import numpy as np

from dataset import ChangeDetectionDataset
from model import UNetModel

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
test_dataset = ChangeDetectionDataset("data/test")

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False
)

# Load model
model = UNetModel().to(device)

model.load_state_dict(
    torch.load("outputs/unet_model.pth", map_location=device)
)

model.eval()

# Metrics
iou_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

with torch.no_grad():

    for images, masks in test_loader:

        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)

        outputs = torch.sigmoid(outputs)

        preds = (outputs > 0.3).float()

        preds_np = preds.cpu().numpy()
        masks_np = masks.cpu().numpy()

        intersection = np.logical_and(preds_np, masks_np).sum()

        union = np.logical_or(preds_np, masks_np).sum()

        iou = intersection / (union + 1e-8)

        precision = intersection / (preds_np.sum() + 1e-8)

        recall = intersection / (masks_np.sum() + 1e-8)

        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

        iou_scores.append(iou)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

# Final metrics
mean_iou = np.mean(iou_scores)
mean_precision = np.mean(precision_scores)
mean_recall = np.mean(recall_scores)
mean_f1 = np.mean(f1_scores)

print(f"IoU: {mean_iou:.4f}")
print(f"Precision: {mean_precision:.4f}")
print(f"Recall: {mean_recall:.4f}")
print(f"F1 Score: {mean_f1:.4f}")