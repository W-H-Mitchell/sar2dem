import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Fully pre-activated residual block as described in paper"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        
        # Clean shortcut path (no activation or batch norm)
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)
        else:
            self.shortcut = nn.Identity()
        
        # Pre-activated main path (BN -> ReLU -> Conv)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding, bias=False)
    
    def forward(self, x):
        # Clean shortcut
        identity = self.shortcut(x)
        
        # Pre-activated path
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        
        return out + identity


class Im2Height(nn.Module):
    """
    Im2Height network adapted for SAR imagery
    Single channel input -> Single channel output
    Uses element-wise summation for skip connection (not concatenation)
    """
    
    def __init__(self, in_channels=1):  # Single channel SAR input
        super().__init__()
        
        # Encoder (convolutional part)
        self.conv1 = ResidualBlock(in_channels, 64)
        self.conv2 = ResidualBlock(64, 128)
        self.conv3 = ResidualBlock(128, 256)
        self.conv4 = ResidualBlock(256, 512)
        
        # Decoder (deconvolutional part)
        self.deconv1 = ResidualBlock(512, 256)
        self.deconv2 = ResidualBlock(256, 128)
        self.deconv3 = ResidualBlock(128, 64)
        self.deconv4 = ResidualBlock(64, 1)  # Output single channel
        
        # Pooling and unpooling
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, 2)
    
    def forward(self, x):
        # For 3-channel input (e.g., from preprocessing), take mean
        if x.shape[1] == 3:
            x = x.mean(dim=1, keepdim=True)
        
        # Encoder with stored pooling indices
        x1 = self.conv1(x)
        x1_skip = x1.clone()  # Store for skip connection
        x1_pooled, idx1 = self.pool(x1)
        
        x2 = self.conv2(x1_pooled)
        x2_pooled, idx2 = self.pool(x2)
        
        x3 = self.conv3(x2_pooled)
        x3_pooled, idx3 = self.pool(x3)
        
        x4 = self.conv4(x3_pooled)
        x4_pooled, idx4 = self.pool(x4)
        
        # Decoder with unpooling using stored indices
        x = self.unpool(x4_pooled, idx4)
        x = self.deconv1(x)
        
        x = self.unpool(x, idx3)
        x = self.deconv2(x)
        
        x = self.unpool(x, idx2)
        x = self.deconv3(x)
        
        x = self.unpool(x, idx1)
        
        # Skip connection with element-wise summation (as per paper)
        x = x + x1_skip
        
        # Final layer
        x = self.deconv4(x)
        
        return x


def load_im2height_model(model, checkpoint_path, device):
    """Load pretrained weights with various checkpoint formats"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    return model