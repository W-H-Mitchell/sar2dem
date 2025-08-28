import torch
import torch.nn as nn

class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type='reflect', norm_layer=nn.InstanceNorm2d, 
                 use_dropout=False, use_bias=True):
        super().__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, 
                                                use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
            
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class GlobalGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_downsampling=4, n_blocks=9, 
                 norm_layer=nn.InstanceNorm2d, padding_type='reflect'):
        super().__init__()
        activation = nn.ReLU(True)

        model = [nn.ReflectionPad2d(3), 
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), 
                 norm_layer(ngf), activation]
        
        # downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, 
                               stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        # resnet blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, 
                                 norm_layer=norm_layer, use_dropout=False, use_bias=True)]
        
        # upsample
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), 
                                         kernel_size=3, stride=2, padding=1, output_padding=1),
                      norm_layer(int(ngf * mult / 2)), activation]
        
        model += [nn.ReflectionPad2d(3), 
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), 
                  nn.Tanh()]
        
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


def load_pix2pixhd_model(checkpoint_path, device):
    """load pix2pixHD generator model"""
    print(f"loading pix2pixHD model from {checkpoint_path}")
    
    # create model
    model = GlobalGenerator(input_nc=3, output_nc=3, ngf=64, 
                           n_downsampling=4, n_blocks=9)
    
    # load checkpoint
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    # handle different checkpoint formats
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    
    print("successfully loaded pix2pixHD model")
    return model