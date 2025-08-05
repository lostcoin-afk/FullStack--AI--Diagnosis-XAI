import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import timm

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

### Step 1: MobileNet Feature Extractor ###
class MobileNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        mobilenet = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        self.feature_extractor = mobilenet.features  # Extract convolutional layers

        # Reduce 960 feature maps into 32 feature maps with learnable weights
        self.conv_compression = nn.Conv2d(960, 32, kernel_size=1)  
        self.relu = nn.ReLU()
        
        # Upscale feature map to 224x224 for ViT compatibility
        self.adaptive_pool = nn.AdaptiveAvgPool2d((224, 224))

    def forward(self, x):
        x = self.feature_extractor(x)  # Shape: (Batch, 960, 7, 7)
        x = self.conv_compression(x)   # Reduce to (Batch, 32, 7, 7)
        x = self.relu(x)  # Apply non-linearity
        x = self.adaptive_pool(x)  # Resize to (Batch, 32, 224, 224)
        return x  # Single-channel feature map ready for ViT

### Step 2: Define Hybrid Model with ViT ###
class HybridViT(nn.Module):
    def __init__(self, num_classes=7):  # Updated for 7-class classification
        super().__init__()
        self.mobilenet = MobileNetFeatureExtractor()
        
        # Load Pretrained ViT (expects images of 224x224)
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=num_classes)

        # Modify input channels for ViT to accept a single feature map
        self.vit.patch_embed.proj = nn.Conv2d(32, 768, kernel_size=16, stride=16)

    def forward(self, x):
        x = self.mobilenet(x)  # Extract single feature map from MobileNet
        x = self.vit(x)  # Feed feature map to ViT
        return x

def hybrid_model_return():
    model = HybridViT(num_classes=7).to(device)  # Updated num_classes
    return model

import os

def get_class_labels_from_folder(data_dir="/home/skrrr/Downloads/archive/test/test"):
    folders = sorted(os.listdir(data_dir))
    return {i: folder for i, folder in enumerate(folders)}
