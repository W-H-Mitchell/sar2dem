import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from src.training.losses.loss_functions import SSIMLoss
ssim = SSIMLoss(data_range=1, channel=1, size_average=False)


class Pool(LightningModule):
	def __init__(self, kernel_size=2, stride=2, **kwargs):
		super(Pool, self).__init__()
		self.pool_fn = nn.MaxPool2d(kernel_size, stride, **kwargs)

	def forward(self, x, *args, **kwargs):
		size = x.size()
		x, indices = self.pool_fn(x, **kwargs)
		return x, indices, size


class Unpool(LightningModule):
	def __init__(self, fn, kernel_size=2, stride=2, **kwargs):
		super(Unpool, self).__init__()
		self.pool_fn = nn.MaxUnpool2d(kernel_size, stride, **kwargs)

	def forward(self, x, indices, output_size, *args, **kwargs):
		return self.pool_fn(x, indices=indices, output_size=output_size, *args, **kwargs)

class Block(LightningModule):
    """ A Block performs three rounds of conv, batchnorm, relu
    """
    def __init__(self, fn, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Block, self).__init__()
        
        # If the input and output dimensions differ, match them with a 1x1 convolution in the skip connection
        self.match_dimensions = in_channels != out_channels or stride != 1
        if self.match_dimensions:
            self.identity = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.identity = nn.Identity()

        # First batch normalization and ReLU activation
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        # The first convolution layer
        self.conv1 = fn(in_channels, out_channels, kernel_size, stride, padding, bias=False)

        # Second batch normalization and ReLU activation
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        # The second convolution layer
        self.conv2 = fn(out_channels, out_channels, kernel_size, 1, padding, bias=False)

    def forward(self, x):
        identity = self.identity(x)

        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        out += identity  # Element-wise addition of the shortcut connection
        return out

class Im2Height(LightningModule):
	""" Im2Height Fully Residual Convolutional-Deconvolutional Network
		implementation based on https://arxiv.org/abs/1802.10249
	"""
	def __init__(self):

		super(Im2Height, self).__init__()

        # Modify the first convolution to accept 3-channel (RGB) input
		self.conv1 = Block(nn.Conv2d, 3, 64)
		self.conv2 = Block(nn.Conv2d, 64, 128)
		self.conv3 = Block(nn.Conv2d, 128, 256)
		self.conv4 = Block(nn.Conv2d, 256, 512)

		# Deconvolutions
		self.deconv1 = Block(nn.ConvTranspose2d, 512, 256)
		self.deconv2 = Block(nn.ConvTranspose2d, 256, 128)
		self.deconv3 = Block(nn.ConvTranspose2d, 128, 64)
		self.deconv4 = Block(nn.ConvTranspose2d, 128, 1) 

		self.pool = Pool(2, 2, return_indices=True)
		self.unpool = Unpool(2, 2)
	
	def forward(self, x):
		x = self.conv1(x)
		
		# Residual skip connection
		x_conv_input = x.clone()
		x, indices1, size1 = self.pool(x)
		x, indices2, size2 = self.pool(self.conv2(x))
		x, indices3, size3 = self.pool(self.conv3(x))
		x, indices4, size4 = self.pool(self.conv4(x))

		# Deconvolve
		x = self.unpool(x, indices4, size4)
		x = self.deconv1(x)
		x = self.unpool(x, indices3, size3)
		x = self.deconv2(x)
		x = self.unpool(x, indices2, size2)
		x = self.deconv3(x)
		x = self.unpool(x, indices1, size1)

		# Element-wise summation with residual skip connection
		# x = x + x_conv_input
		x = torch.cat((x, x_conv_input), dim=1)
		x = self.deconv4(x)

		return x


	# lightning implementations
	def training_step(self, batch, batch_idx):
		x, y = batch
		y_pred = self(x)
		l1loss = F.l1_loss(y_pred, y)
		l2loss = F.mse_loss(y_pred, y)
		tensorboard_logs = { 'l1loss': l1loss, 'l2loss': l2loss }
		return { 'loss': l1loss, 'log': tensorboard_logs }
	
	def configure_optimizers(self):
		return torch.optim.Adam(self.parameters(), lr=1e-4)

	# validation
	def validation_step(self, batch, batch_idx):
		x, y = batch
		y_pred = self(x)
		l1loss = F.l1_loss(y_pred, y)
		l2loss = F.mse_loss(y_pred, y)
		ssim_loss = ssim(y_pred, y)
		tensorboard_logs = { 'val_l1loss': l1loss, 'val_l2loss': l2loss, 'val_ssimloss': ssim_loss }

		return tensorboard_logs

	def validation_epoch_end(self, outputs):

		avg_l1loss = torch.stack([x['val_l1loss'] for x in outputs]).mean()
		avg_l2loss = torch.stack([x['val_l2loss'] for x in outputs]).mean()
		avg_ssimloss = torch.stack([x['val_ssimloss'] for x in outputs]).mean()
		tensorboard_logs = { 'val_l1loss': avg_l1loss, 'val_l2loss': avg_l2loss, 'val_ssimloss': avg_ssimloss }

		return { 'val_l1loss': avg_l1loss, 'log': tensorboard_logs }



if __name__ == "__main__":
	net = Im2Height()
	print(net)