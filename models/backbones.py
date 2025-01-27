import torch
import torch.nn as nn
import torchvision.models as models

class CustomResNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.resnet = models.resnet18(pretrained=True)
        self.conv1 = self.resnet.conv1
        self.bn1 = self.resnet.bn1
        self.relu = self.resnet.relu
        self.maxpool = self.resnet.maxpool
        self.layer1 = self.resnet.layer1
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3
        self.layer4 = self.resnet.layer4


        self.scale1 = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0)
        self.scale2 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)
        self.scale3 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
  
    
    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        # x: (N, 3, 256, 256)
        feature_maps = []
        
        x = self.conv1(x)  
        x = self.bn1(x)
        x = self.relu(x)
        feature_maps.append(x) # (N, 64, 128, 128)
        
        x = self.maxpool(x)
        x = self.layer1(x)  # (64, 64, 64, 64)
        feature_maps.append(self.scale1(x))
        
        x = self.layer2(x)  # (N, 256, 32, 32)
        feature_maps.append(self.scale2(x))
        
        x = self.layer3(x)  # (N, 512, 16, 16)
        feature_maps.append(self.scale3(x))
        
        x = self.layer4(x)  # (N, 512, 4, 4)
        feature_maps.append(x)
        
        feature_maps.append(nn.AdaptiveAvgPool2d((4, 4))(x))  # (N, 512, 4, 4)
        feature_maps.append(nn.AdaptiveAvgPool2d((2, 2))(x))  # (N, 512, 2, 2)
        feature_maps.append(nn.AdaptiveAvgPool2d((1, 1))(x))  # (N, 512, 1, 1)
        
        return feature_maps