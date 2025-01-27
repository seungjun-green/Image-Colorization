import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torchvision.models as models
from models.backbones import CustomResNet




class Downsample(nn.Module):
    """
    Input: (N, C_in, H, W)
    Output: (N, C_out, H/2, W/2)
    """
    def __init__(self, in_channels, out_channels, apply_batchnorm=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        ]
        if apply_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class Upsample(nn.Module):
    """
    Input: (N, C_in, H, W)
    Output: (N, C_out, 2*H, 2*W)
    """
    def __init__(self, in_channels, out_channels, apply_dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ]
        if apply_dropout:
            layers.append(nn.Dropout(0.5))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=1, target_channels=2):
        super().__init__()
        combined_channels = in_channels + target_channels

        self.layer1 = nn.Sequential(
            nn.Conv2d(combined_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.final = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)

    def forward(self, input_image, target_image):
        """
        input: (N, 3, H, W)
        output: (N, )
        Args:
            input_image (N, C_in, H, W)
            target_image: (N, C_out, H, W)
        """
        x = torch.cat([input_image, target_image], dim=1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.final(x)
        return x
    
class UNetGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=2):
        super().__init__()
        self.down1 = Downsample(in_channels, 64, apply_batchnorm=False)
        self.down2 = Downsample(64, 128)
        self.down3 = Downsample(128, 256)
        self.down4 = Downsample(256, 512)
        self.down5 = Downsample(512, 512)
        self.down6 = Downsample(512, 512)
        self.down7 = Downsample(512, 512)
        self.down8 = Downsample(512, 512, apply_batchnorm=False)

        self.up1 = Upsample(512, 512, apply_dropout=True)
        self.up2 = Upsample(1024, 512, apply_dropout=True)
        self.up3 = Upsample(1024, 512, apply_dropout=True)
        self.up4 = Upsample(1024, 512)
        self.up5 = Upsample(1024, 256)
        self.up6 = Upsample(512, 128)
        self.up7 = Upsample(256, 64)

        self.final = nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # encoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        # decoder
        u1 = self.up1(d8)
        u1 = torch.cat([u1, d7], dim=1)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, d6], dim=1)

        u3 = self.up3(u2)
        u3 = torch.cat([u3, d5], dim=1)

        u4 = self.up4(u3)
        u4 = torch.cat([u4, d4], dim=1)

        u5 = self.up5(u4)
        u5 = torch.cat([u5, d3], dim=1)

        u6 = self.up6(u5)
        u6 = torch.cat([u6, d2], dim=1)

        u7 = self.up7(u6)
        u7 = torch.cat([u7, d1], dim=1)

        out = self.final(u7)
        out = self.tanh(out)

        return out
    
class AttenionGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.conv1 = nn.Conv2d(channels,channels // 2, 1)
        self.conv2 = nn.Conv2d(channels // 2, 1, 1)
    
    def forward(self, x):
        '''
        input and output shape are same
        '''
        original_x = x
        # Step1. get the attention mask
        # x: (N, channel, H, W)
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x) # (N, 1, H, W)
        x = torch.sigmoid(x)
        
        # Step2. do the element wise multiplication
        x = original_x * x
        return x

class AttentionUNetGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=2):
        super().__init__()
        self.down1 = Downsample(in_channels, 64, apply_batchnorm=False)
        self.down2 = Downsample(64, 128)
        self.down3 = Downsample(128, 256)
        self.down4 = Downsample(256, 512)
        self.down5 = Downsample(512, 512)
        self.down6 = Downsample(512, 512)
        self.down7 = Downsample(512, 512)
        self.down8 = Downsample(512, 512, apply_batchnorm=False)

        self.up1 = Upsample(512, 512, apply_dropout=True)
        self.up2 = Upsample(1024, 512, apply_dropout=True)
        self.up3 = Upsample(1024, 512, apply_dropout=True)
        self.up4 = Upsample(1024, 512)
        self.up5 = Upsample(1024, 256)
        self.up6 = Upsample(512, 128)
        self.up7 = Upsample(256, 64)

        self.final = nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()
        
        self.ag1 = AttenionGate(1024)
        self.ag2 = AttenionGate(1024)
        self.ag3 = AttenionGate(1024)
        self.ag4 = AttenionGate(1024)
        self.ag5 = AttenionGate(512)
        self.ag6 = AttenionGate(256)
        self.ag7 = AttenionGate(128)

    def forward(self, x):
        # encoder
        d1 = self.down1(x) # (N, 64, H/2, W/2)
        d2 = self.down2(d1) # (N, 128, H/4, W/4)
        d3 = self.down3(d2) # (N, 256, H/8, W/8)
        d4 = self.down4(d3) # (N, 512, H/16, W/16)
        d5 = self.down5(d4) # (N, 512, H/32, W/32)
        d6 = self.down6(d5) # (N, 512, H/64, W/64)
        d7 = self.down7(d6) # (N, 512, H/128, W/128)
        d8 = self.down8(d7) # (N, 512, H/256, W/256)
        

        # decoder
        u1 = self.up1(d8) # (N, 512, H/128, W/128)
        u1 = torch.cat([u1, d7], dim=1) # (N, 1024, H/128, W/128)
        u1 = self.ag1(u1)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, d6], dim=1)
        u2 = self.ag2(u2)

        u3 = self.up3(u2)
        u3 = torch.cat([u3, d5], dim=1)
        u3 = self.ag3(u3)

        u4 = self.up4(u3)
        u4 = torch.cat([u4, d4], dim=1)
        u4 = self.ag4(u4)

        u5 = self.up5(u4)
        u5 = torch.cat([u5, d3], dim=1)
        u5 = self.ag5(u5)

        u6 = self.up6(u5)
        u6 = torch.cat([u6, d2], dim=1)
        u6 = self.ag6(u6)

        u7 = self.up7(u6)
        u7 = torch.cat([u7, d1], dim=1)
        u7 = self.ag7(u7)

        out = self.final(u7)
        out = self.tanh(out)

        return out




class ResNetUNetGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=2):
        super().__init__()
        self.encoder = CustomResNet()
        
        self.up1 = Upsample(512, 512, apply_dropout=True)
        self.up2 = Upsample(1024, 512, apply_dropout=True)
        self.up3 = Upsample(1024, 512, apply_dropout=True)
        self.up4 = Upsample(1024, 512)
        self.up5 = Upsample(1024, 256)
        self.up6 = Upsample(512, 128)
        self.up7 = Upsample(256, 64)

        self.final = nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        ''' Combination of ResNet50 as backbone and Original Decoder of UNet
        Arguments:
            input: 
                x: (N, 3, 256, 256)
        '''
        # x = x.repeat(1, 3, 1, 1) # (N, 3, 256, 256)
        encoder_outputs = self.encoder(x)
        
        
        u1 = self.up1(encoder_outputs[7])
        u1 = torch.cat([u1, encoder_outputs[6]], dim=1)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, encoder_outputs[5]], dim=1)

        u3 = self.up3(u2)
        u3 = torch.cat([u3, encoder_outputs[4]], dim=1)

        u4 = self.up4(u3)
        u4 = torch.cat([u4, encoder_outputs[3]], dim=1)

        u5 = self.up5(u4)
        u5 = torch.cat([u5, encoder_outputs[2]], dim=1)

        u6 = self.up6(u5)
        u6 = torch.cat([u6, encoder_outputs[1]], dim=1)

        u7 = self.up7(u6)
        u7 = torch.cat([u7, encoder_outputs[0]], dim=1)

        out = self.final(u7)
        out = self.tanh(out)

        return out