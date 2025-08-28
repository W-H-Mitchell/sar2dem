import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    """Residual block matching the Lightning implementation"""
    
    def __init__(self, fn, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        
        # Match dimensions if needed
        self.match_dimensions = in_channels != out_channels or stride != 1
        if self.match_dimensions:
            self.identity = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.identity = nn.Identity()
        
        # Pre-activation path
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = fn(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = fn(out_channels, out_channels, kernel_size, 1, padding, bias=False)
    
    def forward(self, x):
        identity = self.identity(x)
        
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        
        out += identity
        return out


class Im2Height(nn.Module):
    """
    Im2Height network matching the trained Lightning model
    Uses 3-channel input and concatenation for skip connections
    """
    
    def __init__(self):
        super().__init__()
        
        # Encoder - 3 channel input (RGB)
        self.conv1 = Block(nn.Conv2d, 3, 64)
        self.conv2 = Block(nn.Conv2d, 64, 128)
        self.conv3 = Block(nn.Conv2d, 128, 256)
        self.conv4 = Block(nn.Conv2d, 256, 512)
        
        # Decoder
        self.deconv1 = Block(nn.ConvTranspose2d, 512, 256)
        self.deconv2 = Block(nn.ConvTranspose2d, 256, 128)
        self.deconv3 = Block(nn.ConvTranspose2d, 128, 64)
        self.deconv4 = Block(nn.ConvTranspose2d, 128, 1)  # 128 input from concatenation
        
        # Pooling/Unpooling
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, 2)
    
    def forward(self, x):
        # Encoder
        x = self.conv1(x)
        x_conv_input = x.clone()  # Save for skip connection
        
        x, indices1 = self.pool(x)
        x = self.conv2(x)
        x, indices2 = self.pool(x)
        x = self.conv3(x)
        x, indices3 = self.pool(x)
        x = self.conv4(x)
        x, indices4 = self.pool(x)
        
        # Decoder
        x = self.unpool(x, indices4)
        x = self.deconv1(x)
        x = self.unpool(x, indices3)
        x = self.deconv2(x)
        x = self.unpool(x, indices2)
        x = self.deconv3(x)
        x = self.unpool(x, indices1)
        
        # Concatenate skip connection (not element-wise sum)
        x = torch.cat((x, x_conv_input), dim=1)  # Results in 128 channels
        x = self.deconv4(x)
        
        return x


def load_im2height_model(model, checkpoint_path, device):
    """Load pretrained weights"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
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
    model.to(device).eval()
    return model