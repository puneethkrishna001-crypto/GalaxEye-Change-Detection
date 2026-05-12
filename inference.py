import torch
import matplotlib.pyplot as plt

from dataset import ChangeDetectionDataset
from model import UNetModel

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
dataset = ChangeDetectionDataset("data/test")

# Load sample
image, mask = dataset[0]

# Add batch dimension
image_batch = image.unsqueeze(0).to(device)

# Load model
model = UNetModel().to(device)

model.load_state_dict(
    torch.load("outputs/unet_model.pth", map_location=device)
)

model.eval()

# Prediction
with torch.no_grad():

    output = model(image_batch)

    output = torch.sigmoid(output)

    pred_mask = (output > 0.5).float()

# Convert to numpy
image_np = image.numpy()

mask_np = mask.squeeze().numpy()

pred_np = pred_mask.squeeze().cpu().numpy()

# EO image
eo = image_np[:3].transpose(1,2,0)

# SAR image
sar = image_np[3]

# Plot
fig, ax = plt.subplots(1,4, figsize=(16,4))

ax[0].imshow(eo)
ax[0].set_title("EO Image")

ax[1].imshow(sar, cmap='gray')
ax[1].set_title("SAR Image")

ax[2].imshow(mask_np, cmap='gray')
ax[2].set_title("Ground Truth")

ax[3].imshow(pred_np, cmap='gray')
ax[3].set_title("Predicted Mask")

plt.tight_layout()

plt.savefig("outputs/prediction.png")

plt.show()