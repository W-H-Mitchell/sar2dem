import torch
import torch.nn as nn

def setupMidasFabric(init_mode="pretrained"):
    """
    Setup MiDaS with different initialization modes.
    
    Args:
        init_mode (str): Initialization mode - "pretrained", "from_scratch", or "decoder_from_scratch"
    """
    model_type = "DPT_Hybrid"
    
    # load model with or without pretrained weights
    if init_mode == "from_scratch":
        print("Loading MiDaS architecture without pretrained weights...")
        midas = torch.hub.load("intel-isl/MiDaS",
                               model_type,
                               force_reload=False,
                               trust_repo=True,
                               skip_validation=True,
                               pretrained=False)
    else:
        print(f"Loading MiDaS with pretrained weights (mode: {init_mode})...")
        midas = torch.hub.load("intel-isl/MiDaS",
                               model_type,
                               force_reload=False,
                               trust_repo=True,
                               skip_validation=True)
    
    # Enable gradients for all parameters
    def _unfreeze_layers(m):
        for param in m.parameters():
            param.requires_grad = True
        return m
    
    midas = _unfreeze_layers(midas)
    
    # Apply initialization based on mode
    if init_mode == "from_scratch":
        print("Initializing all weights from scratch...")
        _initialize_weights_from_scratch(midas)
    elif init_mode == "decoder_from_scratch":
        print("Reinitializing decoder (scratch) layers only...")
        _initialize_decoder_from_scratch(midas)
    elif init_mode == "pretrained":
        print("Keeping all pretrained weights, reinitializing output conv...")
        _random_init_output_conv(midas)
    else:
        raise ValueError(f"Unknown init_mode: {init_mode}. Choose from: pretrained, from_scratch, decoder_from_scratch")
    
    print(f"MiDaS initialization complete (mode: {init_mode})")
    return midas


def _initialize_weights_from_scratch(model):
    """Initialize all model weights from scratch."""
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            torch.nn.init.ones_(m.weight)
            torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            torch.nn.init.ones_(m.weight)
            torch.nn.init.zeros_(m.bias)
        elif hasattr(m, 'weight') and hasattr(m, 'bias'):
            # Generic initialization for other layer types
            if len(m.weight.shape) >= 2:
                torch.nn.init.xavier_uniform_(m.weight)
            else:
                torch.nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    
    model.apply(init_weights)
    return model


def _initialize_decoder_from_scratch(model):
    """Initialize only decoder (scratch) weights, keep pretrained encoder."""
    _random_init_output_conv(model)
    
    #  reinitialize other decoder layers
    if hasattr(model, 'scratch'):
        print("Reinitializing scratch (decoder) module...")
        
        def init_decoder_weights(m):
            if list(m.children()):
                return
                
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                if hasattr(m, 'weight') and m.weight is not None:
                    torch.nn.init.ones_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        
        model.scratch.apply(init_decoder_weights)
    
    return model


def _random_init_output_conv(m):
    """Initialize output convolution layers (original behavior)."""
    for name, param in m.scratch.named_parameters():
        param.requires_grad = True
        if 'weight' in name:
            torch.nn.init.normal_(param.data, 0.0, 0.02)
        if 'bias' in name:
            torch.nn.init.constant_(param.data, 0.0)
    return m


def setupMidasFabricAlternative(init_mode="pretrained"):
    """
    Alternative setup with adaptive normalization layer.
    """
    class AdaptiveMiDaS(nn.Module):
        def __init__(self, init_mode="pretrained"):
            super().__init__()
            
            # Load base MiDaS with specified init mode
            self.midas = setupMidasFabric(init_mode=init_mode)
            
            # Adaptive normalization layer
            self.norm_scale = nn.Parameter(torch.tensor(0.001))
            self.norm_shift = nn.Parameter(torch.tensor(0.0))
            
            print(f"Created AdaptiveMiDaS with {init_mode} initialization:")
            print("  Initial norm_scale: 0.001 (will adapt to data)")
            print("  Initial norm_shift: 0.0 (will learn from data)")
            
        def forward(self, x):
            depth = self.midas(x)
            depth = depth * self.norm_scale + self.norm_shift
            return depth
        
        @property
        def scratch(self):
            return self.midas.scratch
            
        @property
        def pretrained(self):
            return self.midas.pretrained
    
    return AdaptiveMiDaS(init_mode=init_mode)


def loadModelFabric(path, init_mode=None):
    """
    Load a saved model checkpoint.
    
    Args:
        path: Path to checkpoint
        init_mode: If provided, reinitialize model with this mode after loading architecture
    """
    # first create model with default initialization
    midas = setupMidasFabric(init_mode="pretrained")
    
    # load checkpoint
    checkpoint = torch.load(path, map_location='cpu')
    print("Checkpoint keys:", checkpoint.keys())
    
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model_state = checkpoint["model"]
        if hasattr(model_state, 'state_dict'):
            midas.load_state_dict(model_state.state_dict())
        else:
            midas.load_state_dict(model_state)
    else:
        midas.load_state_dict(checkpoint)
    
    # if init_mode is specified, reinitialize accordingly
    if init_mode and init_mode != "pretrained":
        print(f"Reinitializing model with mode: {init_mode}")
        if init_mode == "from_scratch":
            _initialize_weights_from_scratch(midas)
        elif init_mode == "decoder_from_scratch":
            _initialize_decoder_from_scratch(midas)
    
    return midas
