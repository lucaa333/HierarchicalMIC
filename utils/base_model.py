"""
Base 3D CNN architecture for medical image classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Base3DCNN(nn.Module):
    """
    Base 3D Convolutional Neural Network for medical image classification.

    This architecture uses 3D convolutions to process volumetric medical images
    and extract spatial features across all three dimensions.

    Args:
        in_channels (int): Number of input channels (default: 1 for grayscale)
        num_classes (int): Number of output classes
        dropout_rate (float): Dropout rate for regularization (default: 0.3)
    """

    def __init__(self, in_channels=1, num_classes=11, dropout_rate=0.3):
        super(Base3DCNN, self).__init__()

        # First convolutional block
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),  # 28x28x28 -> 14x14x14
        )

        # Second convolutional block
        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),  # 14x14x14 -> 7x7x7
        )

        # Third convolutional block
        self.conv3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),  # 7x7x7 -> 3x3x3
        )

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, depth, height, width)

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

    def extract_features(self, x):
        """
        Extract features before final classification layer.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Feature tensor
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        return x


class ResidualBlock3D(nn.Module):
    """
    3D Residual Block for improved gradient flow and deeper networks.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock3D, self).__init__()

        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm3d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Enhanced3DCNN(nn.Module):
    """
    Enhanced 3D CNN with residual connections for better performance.
    """

    def __init__(self, in_channels=1, num_classes=11, dropout_rate=0.3):
        super(Enhanced3DCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        self.layer1 = self._make_layer(32, 32, 2, stride=1)
        self.layer2 = self._make_layer(32, 64, 2, stride=2)
        self.layer3 = self._make_layer(64, 128, 2, stride=2)

        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)

        self.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes),
        )

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock3D(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock3D(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
