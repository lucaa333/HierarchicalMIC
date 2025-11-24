"""
Stage 2: Fine-level pathology classifier
Region-specific classifiers for disease detection
"""

import torch
import torch.nn as nn
from .base_model import Base3DCNN, Enhanced3DCNN
from .cnn_3d_models import get_3d_model


class FinePathologyClassifier(nn.Module):
    """
    Stage 2 classifier for pathology identification within anatomical regions.
    
    This creates specialized classifiers for each anatomical region to identify
    specific diseases or abnormalities, following the coarse localization.
    
    Args:
        region_name (str): Anatomical region ('brain', 'abdomen', 'chest')
        num_pathologies (int): Number of pathology classes for this region
        architecture (str): Model architecture ('base', 'enhanced', 'resnet18_3d', 
                           'resnet34_3d', 'resnet50_3d', 'densenet121_3d', 'efficientnet3d_b0')
        dropout_rate (float): Dropout rate
    """
    
    def __init__(self, region_name, num_pathologies, architecture='resnet18_3d', dropout_rate=0.3):
        super(FinePathologyClassifier, self).__init__()
        
        self.region_name = region_name
        self.num_pathologies = num_pathologies
        self.architecture = architecture
        
        # Select model architecture
        if architecture == 'enhanced':
            self.model = Enhanced3DCNN(
                in_channels=1,
                num_classes=num_pathologies,
                dropout_rate=dropout_rate
            )
        elif architecture == 'base':
            self.model = Base3DCNN(
                in_channels=1,
                num_classes=num_pathologies,
                dropout_rate=dropout_rate
            )
        else:
            # Use advanced 3D models (ResNet, DenseNet, EfficientNet)
            self.model = get_3d_model(
                model_name=architecture,
                num_classes=num_pathologies,
                dropout_rate=dropout_rate
            )
    
    def forward(self, x):
        """
        Forward pass for pathology classification.
        
        Args:
            x (torch.Tensor): Input 3D medical images from specific region
        
        Returns:
            torch.Tensor: Logits for each pathology class
        """
        return self.model(x)
    
    def predict_pathology(self, x):
        """
        Predict pathology with confidence scores.
        
        Args:
            x (torch.Tensor): Input images
        
        Returns:
            tuple: (pathology_indices, confidence_scores)
        """
        logits = self.forward(x)
        probabilities = torch.softmax(logits, dim=1)
        confidence_scores, pathology_indices = torch.max(probabilities, dim=1)
        
        return pathology_indices, confidence_scores


class RegionSpecificPathologyNetwork(nn.Module):
    """
    Container for multiple region-specific pathology classifiers.
    
    Manages separate specialized models for different anatomical regions,
    allowing independent training and inference for each region.
    """
    
    def __init__(self, region_configs, architecture='resnet18_3d', dropout_rate=0.3):
        """
        Args:
            region_configs (dict): Dict mapping region names to number of pathologies
                Example: {'brain': 5, 'abdomen': 8, 'chest': 6}
            architecture (str): Architecture type ('base', 'enhanced', 'resnet18_3d', etc.)
            dropout_rate (float): Dropout rate
        """
        super(RegionSpecificPathologyNetwork, self).__init__()
        
        self.region_configs = region_configs
        self.classifiers = nn.ModuleDict()
        
        for region_name, num_pathologies in region_configs.items():
            self.classifiers[region_name] = FinePathologyClassifier(
                region_name=region_name,
                num_pathologies=num_pathologies,
                architecture=architecture,
                dropout_rate=dropout_rate
            )
    
    def forward(self, x, region_name):
        """
        Forward pass through region-specific classifier.
        
        Args:
            x (torch.Tensor): Input images
            region_name (str): Name of the anatomical region
        
        Returns:
            torch.Tensor: Pathology logits
        """
        if region_name not in self.classifiers:
            raise ValueError(f"Unknown region: {region_name}")
        
        return self.classifiers[region_name](x)
    
    def get_classifier(self, region_name):
        """Get classifier for specific region."""
        return self.classifiers.get(region_name)