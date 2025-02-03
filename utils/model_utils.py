from models.models import *

def initialize_weights(model):
    """ Initializes model weights using Xavier Normal Initialization (as per pix2pix paper). """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1) 
            nn.init.constant_(m.bias, 0)
            

def load_generator(model_type):
    ''' Functions to load generator
    
    Argument:
        model_type(str): config.generator
    '''
    ## load generator
    if model_type == "UNet":
        return UNetGenerator()
    elif model_type == "ResNetUNet":
        return ResNetUNetGenerator()
    elif model_type == "AttentionUNet":
        return AttentionUNetGenerator()
    else:
        raise ValueError("Unsupproted generator type")
    
def load_discriminator(model_type):
    ''' Functions to load discriminator
    Argument:
        model_type: config.discriminator
    '''
    ## load generator
    if model_type == "PatchGANDiscriminator":
        return PatchGANDiscriminator()
    elif model_type == "some other options":
        return PatchGANDiscriminator()
    else:
        raise ValueError("Unsupproted discriminator type")