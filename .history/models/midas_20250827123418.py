import torch
import torch.nn as nn

def setupMidasFabric():
    """
    Setup MiDaS with different initialization modes.
    
    Args:
        init_mode (str): Initialization mode - "pretrained", "from_scratch", or "decoder_from_scratch"
    """
    model_type = "DPT_Hybrid"
    midas = torch.hub.load("intel-isl/MiDaS",
                            model_type,
                            force_reload=False,
                            trust_repo=True,
                            skip_validation=True)
    

    def _unfreeze_layers(m):
        for param in m.parameters():
            param.requires_grad = True
        return m
    
    def _random_init_output_conv(m):
        """Initialize output convolution layers."""
        for name, param in m.scratch.named_parameters():
            param.requires_grad = True
            if 'weight' in name:
                torch.nn.init.normal_(param.data, 0.0, 0.02)
            if 'bias' in name:
                torch.nn.init.constant_(param.data, 0.0)
        return m
    
    midas = _unfreeze_layers(midas)
    midas = _random_init_output_conv(midas)
    
    return midas


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
