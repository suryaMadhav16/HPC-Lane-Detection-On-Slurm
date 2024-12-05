import torch
import torch.nn as nn
import torchvision.models as models
from .attention import CoordAttention

class LaneDetectionModel(nn.Module):
    """
    Lane Detection Model with ResNet backbone and Coordinate Attention
    """
    def __init__(self, 
                 num_classes: int = 2,
                 backbone: str = 'resnet50',
                 pretrained: bool = True):
        """
        Initialize the Lane Detection Model.
        
        Args:
            num_classes (int): Number of output classes
            backbone (str): Backbone model name
            pretrained (bool): Whether to use pretrained backbone
        """
        super(LaneDetectionModel, self).__init__()
        
        # Initialize backbone
        if backbone == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
            self.encoder = nn.Sequential(*list(resnet.children())[:-2])
            encoder_channels = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Initialize attention
        self.coord_att = CoordAttention(encoder_channels, encoder_channels)
        
        # Decoder layers
        self.up1 = nn.Sequential(
            nn.Conv2d(encoder_channels, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        )
        
        self.up2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        )
        
        self.up3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        )
        
        # Final classification layer
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Output segmentation map
        """
        # Encoder
        features = self.encoder(x)
        
        # Apply attention
        features = self.coord_att(features)
        
        # Decoder
        x = self.up1(features)
        x = self.up2(x)
        x = self.up3(x)
        
        # Final prediction
        x = self.final_conv(x)
        
        return x

    def get_backbone_parameters(self):
        """Get backbone parameters for separate optimization"""
        return self.encoder.parameters()

    def get_decoder_parameters(self):
        """Get decoder parameters for separate optimization"""
        modules = [self.up1, self.up2, self.up3, self.final_conv]
        for module in modules:
            for param in module.parameters():
                yield param