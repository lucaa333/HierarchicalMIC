"""
Production-ready 3D CNN architectures for medical image classification.

Includes:
- 3D ResNet (18/34/50) - Best for classification
- 3D DenseNet (121) - Excellent feature reuse
- 3D EfficientNet - Efficient parameter usage
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# 3D ResNet Architecture (ResNet-18, ResNet-34, ResNet-50)
# ============================================================================


class BasicBlock3D(nn.Module):
    """Basic 3D residual block for ResNet-18/34"""

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        # Use track_running_stats=True but handle batch_size=1 gracefully
        self.bn1 = nn.BatchNorm3d(out_channels, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(out_channels, track_running_stats=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck3D(nn.Module):
    """Bottleneck 3D residual block for ResNet-50/101/152"""

    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.conv3 = nn.Conv3d(
            out_channels, out_channels * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm3d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet3D(nn.Module):
    """
    3D ResNet for volumetric medical image classification.

    Supports ResNet-18, ResNet-34, ResNet-50 architectures.
    Based on "Deep Residual Learning for Image Recognition" (He et al., 2016)
    adapted for 3D medical imaging.

    Args:
        block: BasicBlock3D or Bottleneck3D
        layers: List of layer depths [layer1, layer2, layer3, layer4]
        num_classes: Number of output classes
        dropout_rate: Dropout rate for regularization
    """

    def __init__(self, block, layers, num_classes=11, dropout_rate=0.3):
        super().__init__()
        self.in_channels = 64

        # Initial convolution
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Classification head
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm3d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Handle batch_size=1 during training (for hierarchical routing)
        # Temporarily set BatchNorm to eval mode if batch size is 1
        single_sample = (x.size(0) == 1) and self.training
        if single_sample:
            self.eval()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        # Restore training mode if we temporarily switched
        if single_sample:
            self.train()

        return x


def resnet18_3d(num_classes=11, dropout_rate=0.3):
    """3D ResNet-18"""
    return ResNet3D(BasicBlock3D, [2, 2, 2, 2], num_classes, dropout_rate)


def resnet34_3d(num_classes=11, dropout_rate=0.3):
    """3D ResNet-34"""
    return ResNet3D(BasicBlock3D, [3, 4, 6, 3], num_classes, dropout_rate)


def resnet50_3d(num_classes=11, dropout_rate=0.3):
    """3D ResNet-50"""
    return ResNet3D(Bottleneck3D, [3, 4, 6, 3], num_classes, dropout_rate)


# ============================================================================
# 3D DenseNet Architecture
# ============================================================================


class DenseLayer3D(nn.Module):
    """Single dense layer in DenseNet"""

    def __init__(self, in_channels, growth_rate, bn_size=4, dropout_rate=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(
            in_channels, bn_size * growth_rate, kernel_size=1, bias=False
        )

        self.bn2 = nn.BatchNorm3d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            bn_size * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False
        )

        self.dropout_rate = dropout_rate
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.conv2(self.relu2(self.bn2(out)))

        if self.dropout_rate > 0:
            out = self.dropout(out)

        return torch.cat([x, out], 1)


class DenseBlock3D(nn.Module):
    """Dense block with multiple dense layers"""

    def __init__(
        self, num_layers, in_channels, growth_rate, bn_size=4, dropout_rate=0.0
    ):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(
                DenseLayer3D(
                    in_channels + i * growth_rate, growth_rate, bn_size, dropout_rate
                )
            )
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Transition3D(nn.Module):
    """Transition layer between dense blocks"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn = nn.BatchNorm3d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.conv(self.relu(self.bn(x)))
        out = self.pool(out)
        return out


class DenseNet3D(nn.Module):
    """
    3D DenseNet for volumetric medical image classification.

    Based on "Densely Connected Convolutional Networks" (Huang et al., 2017)
    adapted for 3D medical imaging.

    Args:
        growth_rate: Number of filters added per layer
        block_config: Number of layers in each dense block
        num_init_features: Number of filters in first convolution
        bn_size: Bottleneck size multiplier
        dropout_rate: Dropout rate
        num_classes: Number of output classes
    """

    def __init__(
        self,
        growth_rate=32,
        block_config=(6, 12, 24, 16),
        num_init_features=64,
        bn_size=4,
        dropout_rate=0.3,
        num_classes=11,
    ):
        super().__init__()

        # Initial convolution
        self.features = nn.Sequential(
            nn.Conv3d(
                1, num_init_features, kernel_size=7, stride=2, padding=3, bias=False
            ),
            nn.BatchNorm3d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )

        # Dense blocks
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlock3D(
                num_layers=num_layers,
                in_channels=num_features,
                growth_rate=growth_rate,
                bn_size=bn_size,
                dropout_rate=0.0,  # No dropout in dense layers
            )
            self.features.add_module(f"denseblock{i + 1}", block)
            num_features += num_layers * growth_rate

            if i != len(block_config) - 1:
                trans = Transition3D(num_features, num_features // 2)
                self.features.add_module(f"transition{i + 1}", trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module("norm5", nn.BatchNorm3d(num_features))
        self.features.add_module("relu5", nn.ReLU(inplace=True))

        # Classification head
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(num_features, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = self.avgpool(features)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        out = self.classifier(out)
        return out


def densenet121_3d(num_classes=11, dropout_rate=0.3):
    """
    3D DenseNet-121 adapted for small medical images (28x28x28).
    Uses fewer dense blocks to prevent over-downsampling.
    """
    return DenseNet3D(
        growth_rate=32,
        block_config=(6, 12, 16),  # 3 blocks instead of 4 for 28x28x28 input
        num_init_features=64,
        dropout_rate=dropout_rate,
        num_classes=num_classes,
    )


# ============================================================================
# 3D EfficientNet-Inspired Architecture
# ============================================================================


class MBConv3D(nn.Module):
    """Mobile Inverted Bottleneck Convolution for 3D"""

    def __init__(
        self, in_channels, out_channels, expand_ratio=6, stride=1, dropout_rate=0.2
    ):
        super().__init__()
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels

        hidden_dim = in_channels * expand_ratio

        layers = []
        if expand_ratio != 1:
            # Expansion
            layers.extend(
                [
                    nn.Conv3d(in_channels, hidden_dim, 1, bias=False),
                    nn.BatchNorm3d(hidden_dim),
                    nn.ReLU6(inplace=True),
                ]
            )

        # Depthwise
        layers.extend(
            [
                nn.Conv3d(
                    hidden_dim,
                    hidden_dim,
                    3,
                    stride=stride,
                    padding=1,
                    groups=hidden_dim,
                    bias=False,
                ),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace=True),
            ]
        )

        # Projection
        layers.extend(
            [
                nn.Conv3d(hidden_dim, out_channels, 1, bias=False),
                nn.BatchNorm3d(out_channels),
            ]
        )

        self.conv = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, x):
        if self.use_residual:
            out = self.conv(x)
            if self.dropout is not None:
                out = self.dropout(out)
            return x + out
        else:
            return self.conv(x)


class EfficientNet3D(nn.Module):
    """
    3D EfficientNet-inspired architecture for medical imaging.

    Lightweight and efficient design with inverted residual blocks.
    Optimized for medical image classification with limited data.

    Args:
        num_classes: Number of output classes
        width_mult: Width multiplier for channels
        dropout_rate: Dropout rate
    """

    def __init__(self, num_classes=11, width_mult=1.0, dropout_rate=0.3):
        super().__init__()

        # Initial convolution
        input_channel = int(32 * width_mult)
        self.features = nn.Sequential(
            nn.Conv3d(1, input_channel, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(input_channel),
            nn.ReLU6(inplace=True),
        )

        # MBConv blocks
        # [expand_ratio, channels, num_blocks, stride]
        settings = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 40, 2, 2],
            [6, 80, 3, 2],
            [6, 112, 3, 1],
            [6, 192, 1, 1],
        ]

        features = []
        for expand_ratio, c, n, s in settings:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    MBConv3D(
                        input_channel,
                        output_channel,
                        expand_ratio,
                        stride,
                        dropout_rate=0.2,
                    )
                )
                input_channel = output_channel

        self.features = nn.Sequential(self.features, *features)

        # Final layers
        final_channel = int(1280 * width_mult)
        self.conv_final = nn.Sequential(
            nn.Conv3d(input_channel, final_channel, 1, bias=False),
            nn.BatchNorm3d(final_channel),
            nn.ReLU6(inplace=True),
        )

        # Classification head
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(final_channel, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.conv_final(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


def efficientnet3d_b0(num_classes=11, dropout_rate=0.3):
    """3D EfficientNet-B0 (baseline)"""
    return EfficientNet3D(num_classes, width_mult=1.0, dropout_rate=dropout_rate)


# ============================================================================
# Factory Function
# ============================================================================


def get_3d_model(model_name, num_classes=11, dropout_rate=0.3):
    """
    Factory function to create 3D models.

    Args:
        model_name: One of ['resnet18_3d', 'resnet34_3d', 'resnet50_3d',
                    'densenet121_3d', 'efficientnet3d_b0']
        num_classes: Number of output classes
        dropout_rate: Dropout rate

    Returns:
        Model instance
    """
    models_dict = {
        "resnet18_3d": resnet18_3d,
        "resnet34_3d": resnet34_3d,
        "resnet50_3d": resnet50_3d,
        "densenet121_3d": densenet121_3d,
        "efficientnet3d_b0": efficientnet3d_b0,
    }

    if model_name.lower() not in models_dict:
        raise ValueError(
            f"Unknown model: {model_name}. Choose from {list(models_dict.keys())}"
        )

    model_fn = models_dict[model_name.lower()]
    return model_fn(num_classes=num_classes, dropout_rate=dropout_rate)
