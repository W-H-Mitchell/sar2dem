import torch
import torch.nn as nn

class Pool(nn.Module):
    def __init__(self, kernel_size=2, stride=2, return_indices=True):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size, stride, return_indices=return_indices)
    
    def forward(self, x):
        return self.pool(x)

class Unpool(nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.unpool = nn.MaxUnpool2d(kernel_size, stride)
    
    def forward(self, x, indices, output_size):
        return self.unpool(x, indices, output_size)

class Block(nn.Module):
    def __init__(self, fn, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        
        self.match_dimensions = in_channels != out_channels or stride != 1
        if self.match_dimensions:
            self.identity = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.identity = nn.Identity()

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
    def __init__(self):
        super().__init__()
        
        self.conv1 = Block(nn.Conv2d, 3, 64)
        self.conv2 = Block(nn.Conv2d, 64, 128)
        self.conv3 = Block(nn.Conv2d, 128, 256)
        self.conv4 = Block(nn.Conv2d, 256, 512)
        
        self.deconv1 = Block(nn.ConvTranspose2d, 512, 256)
        self.deconv2 = Block(nn.ConvTranspose2d, 256, 128)
        self.deconv3 = Block(nn.ConvTranspose2d, 128, 64)
        self.deconv4 = Block(nn.ConvTranspose2d, 128, 1)
        
        self.pool = Pool(2, 2, return_indices=True)
        self.unpool = Unpool(2, 2)
    
    def forward(self, x):
        x = self.conv1(x)
        x_conv_input = x.clone()
        
        x, indices1, size1 = self.pool(x)
        x = self.conv2(x)
        x, indices2, size2 = self.pool(x)
        x = self.conv3(x)
        x, indices3, size3 = self.pool(x)
        x = self.conv4(x)
        x, indices4, size4 = self.pool(x)
        
        x = self.unpool(x, indices4, size4)
        x = self.deconv1(x)
        x = self.unpool(x, indices3, size3)
        x = self.deconv2(x)
        x = self.unpool(x, indices2, size2)
        x = self.deconv3(x)
        x = self.unpool(x, indices1, size1)
        
        x = torch.cat((x, x_conv_input), dim=1)
        x = self.deconv4(x)
        return x