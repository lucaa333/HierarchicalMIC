"""
Stage 3: Subtype classifier for detailed disease categorization
"""

import torch
import torch.nn as nn
from .base_model import Base3DCNN
from .cnn_3d_models import get_3d_model


class SubtypeClassifier(nn.Module):
    """
    Stage 3 classifier for disease subtype identification.
    
    This classifier provides fine-grained categorization of disease subtypes
    within each pathology category, representing the most detailed level
    of the hierarchical classification.
    
    Args:
        pathology_name (str): Name of the pathology
        num_subtypes (int): Number of subtype classes
        architecture (str): Model architecture ('base', 'resnet18_3d', 'resnet34_3d', 
                           'resnet50_3d', 'densenet121_3d', 'efficientnet3d_b0')
        dropout_rate (float): Dropout rate
    """
    
    def __init__(self, pathology_name, num_subtypes, architecture='resnet18_3d', dropout_rate=0.3):
        super(SubtypeClassifier, self).__init__()
        
        self.pathology_name = pathology_name
        self.num_subtypes = num_subtypes
        self.architecture = architecture
        
        # Select model architecture
        if architecture == 'base':
            self.model = Base3DCNN(
                in_channels=1,
                num_classes=num_subtypes,
                dropout_rate=dropout_rate
            )
        else:
            self.model = get_3d_model(
                model_name=architecture,
                num_classes=num_subtypes,
                dropout_rate=dropout_rate
            )
    
    def forward(self, x):
        """
        Forward pass for subtype classification.
        
        Args:
            x (torch.Tensor): Input 3D medical images
        
        Returns:
            torch.Tensor: Logits for each subtype
        """
        return self.model(x)
    
    def predict_subtype(self, x):
        """
        Predict subtype with confidence scores.
        
        Args:
            x (torch.Tensor): Input images
        
        Returns:
            tuple: (subtype_indices, confidence_scores)
        """
        logits = self.forward(x)
        probabilities = torch.softmax(logits, dim=1)
        confidence_scores, subtype_indices = torch.max(probabilities, dim=1)
        
        return subtype_indices, confidence_scores


class HierarchicalSubtypeNetwork(nn.Module):
    """
    Container for multiple pathology-specific subtype classifiers.
    
    Manages the third stage of hierarchical classification with specialized
    models for identifying subtypes within each pathology category.
    """
    
    def __init__(self, subtype_configs, architecture='resnet18_3d', dropout_rate=0.3):
        """
        Args:
            subtype_configs (dict): Nested dict mapping regions -> pathologies -> num_subtypes
                Example: {
                    'brain': {
                        'tumor': 4,
                        'hemorrhage': 3
                    },
                    'chest': {
                        'pneumonia': 2,
                        'tumor': 3
                    }
                }
            architecture (str): Model architecture for all subtype classifiers
            dropout_rate (float): Dropout rate
        """
        super(HierarchicalSubtypeNetwork, self).__init__()
        
        self.subtype_configs = subtype_configs
        self.architecture = architecture
        self.classifiers = nn.ModuleDict()
        
        # Create classifiers for each pathology
        for region_name, pathologies in subtype_configs.items():
            for pathology_name, num_subtypes in pathologies.items():
                key = f"{region_name}_{pathology_name}"
                self.classifiers[key] = SubtypeClassifier(
                    pathology_name=pathology_name,
                    num_subtypes=num_subtypes,
                    architecture=architecture,
                    dropout_rate=dropout_rate
                )
    
    def forward(self, x, region_name, pathology_name):
        """
        Forward pass through pathology-specific subtype classifier.
        
        Args:
            x (torch.Tensor): Input images
            region_name (str): Anatomical region name
            pathology_name (str): Pathology name
        
        Returns:
            torch.Tensor: Subtype logits
        """
        key = f"{region_name}_{pathology_name}"
        
        if key not in self.classifiers:
            raise ValueError(f"Unknown pathology: {region_name}/{pathology_name}")
        
        return self.classifiers[key](x)
    
    def get_classifier(self, region_name, pathology_name):
        """Get classifier for specific pathology."""
        key = f"{region_name}_{pathology_name}"
        return self.classifiers.get(key)