import torch
import torch.nn as nn
from torchvision.models import resnet50, resnet152

class ResNetBackbone(nn.Module):
    """
    Custom ResNet backbone for feature extraction.
    Removes the final classification layer and keeps convolutional layers.
    """
    def __init__(self, arch: str = "resnet50", pretrained: bool = True):
        """
        Initializes the ResNet backbone.
        Args:
            arch (str): The ResNet architecture to use ('resnet50' or 'resnet152').
            pretrained (bool): Whether to load pretrained ImageNet weights.
        """
        super(ResNetBackbone, self).__init__()
        if arch == "resnet50":
            resnet = resnet50(pretrained=pretrained)
            self.feature_dim = 2048  
        elif arch == "resnet152":
            resnet = resnet152(pretrained=pretrained)
            self.feature_dim = 2048  
        else:
            raise ValueError("Invalid architecture: Choose 'resnet50' or 'resnet152'")

        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # Remove final FC layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of ResNet.
        Args:
            x (torch.Tensor): Input tensor (batch_size, C, H, W)
        Returns:
            torch.Tensor: Extracted feature maps (batch_size, feature_dim, 1, 1)
        """
        return self.backbone(x)

def get_resnet_backbone(arch: str = "resnet50", pretrained: bool = True) -> nn.Module:
    """
    Function to get a ResNet backbone.
    """
    return ResNetBackbone(arch=arch, pretrained=pretrained)


class RMSNetModel(nn.Module):
    def __init__(self, num_classes: int, resnet_arch: str = "resnet50"):
        """
        Initializes the SoccerNet model.

        Args:
            num_classes (int): Number of classes.
            resnet_arch (str): ResNet backbone ('resnet50' or 'resnet152').
        """
        super(RMSNetModel, self).__init__()

        #ResNet Backbone
        self.backbone = get_resnet_backbone(arch=resnet_arch, pretrained=True)
        num_features = self.backbone.feature_dim  # Adapted based on architecture

        #Temporal 1D Convolution
        self.temporal_conv1 = nn.Conv1d(in_channels=num_features, out_channels=512, kernel_size=9, padding=4)
        self.temporal_conv2 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=9, padding=4)
        self.fc2 = nn.Linear(256, 128)
        
        #Event Classification
        self.fc_class = nn.Linear(128, num_classes)
        #Timestamp regression
        self.fc_t_shift = nn.Linear(128, 1)

    def forward(self, x):
        x = self.backbone(x)  # Extract features from ResNet
        x = x.view(x.size(0), x.size(1), -1)  # Reshape for temporal conv layers
        x = self.temporal_conv1(x)
        x = self.temporal_conv2(x)
        x = x.mean(dim=2)
        features = self.fc2(x)
        class_output = self.fc_class(features)
        time_shift_output = self.fc_t_shift(features)
        return class_output, time_shift_output