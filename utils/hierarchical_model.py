"""
Complete hierarchical classification pipeline
Integrates all three stages: coarse -> fine -> subtype
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from .coarse_classifier import CoarseAnatomicalClassifier
from .fine_classifier import RegionSpecificPathologyNetwork
from .subtype_classifier import HierarchicalSubtypeNetwork
from .cnn_3d_models import get_3d_model


class HierarchicalClassificationModel(nn.Module):
    """
    End-to-end hierarchical medical image classification model.
    
    Three-stage pipeline:
    1. Coarse: Anatomical region (brain, abdomen, chest)
    2. Fine: Region-specific pathology
    3. Subtype: Disease subtype (optional)
    """
    
    def __init__(
        self,
        region_configs: Dict[str, int],
        subtype_configs: Optional[Dict] = None,
        architecture: str = 'base',
        use_subtypes: bool = False,
        dropout_rate: float = 0.3,
        organ_to_region_map: Optional[Dict[int, int]] = None,
        num_total_organs: int = 11,
        region_idx_to_name: Optional[Dict[int, str]] = None,
        coarse_model_type: str = 'resnet18_3d',  # 'base', 'resnet18_3d', 'resnet34_3d', 'resnet50_3d', etc.
        fine_model_type: str = 'resnet18_3d',    # 'base', 'resnet18_3d', 'resnet34_3d', 'resnet50_3d', etc.
    ):
        super(HierarchicalClassificationModel, self).__init__()
        
        self.region_configs = region_configs
        self.use_subtypes = use_subtypes
        self.num_total_organs = num_total_organs
        self.organ_to_region_map = organ_to_region_map
        self.region_idx_to_name = region_idx_to_name
        self.coarse_model_type = coarse_model_type
        self.fine_model_type = fine_model_type
        
        # Build region-to-organs mapping for converting local to global indices
        self._build_organ_mappings()
        
        # Stage 1: Coarse classifier
        self.coarse_classifier = CoarseAnatomicalClassifier(
            architecture=coarse_model_type,
            num_regions=len(region_configs),
            dropout_rate=dropout_rate,
            region_names=region_idx_to_name
        )
        
        # Stage 2: Fine classifiers
        self.fine_classifier = RegionSpecificPathologyNetwork(
            region_configs=region_configs,
            architecture=fine_model_type,
            dropout_rate=dropout_rate
        )
        
        # Stage 3: Subtype classifiers (optional)
        if use_subtypes and subtype_configs:
            self.subtype_classifier = HierarchicalSubtypeNetwork(
                subtype_configs=subtype_configs,
                architecture=fine_model_type,  # Use same architecture as fine level
                dropout_rate=dropout_rate
            )
        else:
            self.subtype_classifier = None
    
    def _build_organ_mappings(self):
        """Build mappings between region-specific and global organ indices."""
        if self.organ_to_region_map is None:
            # No mapping provided, assume identity mapping
            self.region_to_organs = {}
            self.local_to_global = {}
            return
        
        # Group organs by region
        region_idx_to_name = {i: name for i, name in enumerate(self.region_configs.keys())}
        self.region_to_organs = {region_name: [] for region_name in self.region_configs.keys()}
        
        for organ_idx, region_idx in self.organ_to_region_map.items():
            region_name = region_idx_to_name[region_idx]
            self.region_to_organs[region_name].append(organ_idx)
        
        # Sort organs within each region for consistent indexing
        for region_name in self.region_to_organs:
            self.region_to_organs[region_name].sort()
        
        # Build local-to-global mapping: {region_name: {local_idx: global_idx}}
        self.local_to_global = {}
        for region_name, organ_list in self.region_to_organs.items():
            self.local_to_global[region_name] = {local_idx: global_idx 
                                                   for local_idx, global_idx in enumerate(organ_list)}
    
    def forward_coarse(self, x):
        """Stage 1: Predict anatomical region."""
        return self.coarse_classifier(x)
    
    def forward_fine(self, x, region_name):
        """Stage 2: Predict pathology within region."""
        if isinstance(self.fine_classifier, nn.ModuleDict):
            # Using custom models (ModuleDict)
            return self.fine_classifier[region_name](x)
        else:
            # Using default RegionSpecificPathologyNetwork
            return self.fine_classifier(x, region_name)
    
    def forward_subtype(self, x, region_name, pathology_name):
        """Stage 3: Predict disease subtype."""
        if self.subtype_classifier is None:
            raise ValueError("Subtype classification not enabled")
        return self.subtype_classifier(x, region_name, pathology_name)
    
    def forward(self, x, region_indices=None, return_dict=False):
        """
        Complete forward pass.
        
        Args:
            x: Input images
            region_indices: Pre-computed region indices (optional)
            return_dict: If True, return dict with all stages. If False, return final logits only.
        
        Returns:
            torch.Tensor or dict: Final organ logits (default) or dict with all stages
        """
        batch_size = x.size(0)
        
        # Stage 1: Region classification
        region_logits = self.forward_coarse(x)
        region_probs = torch.softmax(region_logits, dim=1)
        confidence, region_predictions = torch.max(region_probs, dim=1)
        
        # Stage 2: Route through fine classifiers based on predicted regions
        # Get region index to name mapping
        region_idx_to_name = {i: name for i, name in enumerate(self.region_configs.keys())}
        
        # Initialize output tensor for all organ classes
        # Shape: (batch_size, num_total_organs)
        organ_logits = torch.full((batch_size, self.num_total_organs), -1e9, 
                                   device=x.device, dtype=torch.float32)
        
        # Group samples by predicted region for batch processing
        region_samples = {region_name: [] for region_name in self.region_configs.keys()}
        region_indices_map = {region_name: [] for region_name in self.region_configs.keys()}
        
        for i in range(batch_size):
            region_idx = region_predictions[i].item()
            region_name = region_idx_to_name[region_idx]
            region_samples[region_name].append(x[i:i+1])
            region_indices_map[region_name].append(i)
        
        # Process each region's samples in batch
        # Temporarily set fine classifiers to eval mode to avoid batch norm issues with small batches
        if isinstance(self.fine_classifier, nn.ModuleDict):
            # Store original training states
            original_states = {}
            for region_name, model in self.fine_classifier.items():
                original_states[region_name] = model.training
                # Set batch norm layers to eval mode while keeping other layers in training mode
                for module in model.modules():
                    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                        module.eval()
        
        for region_name in self.region_configs.keys():
            if len(region_samples[region_name]) > 0:
                # Concatenate samples for this region
                region_batch = torch.cat(region_samples[region_name], dim=0)
                
                # Forward through fine classifier
                local_logits = self.forward_fine(region_batch, region_name)
                
                # Map results back to original batch positions
                for local_idx, batch_idx in enumerate(region_indices_map[region_name]):
                    # Map local organ indices to global organ indices
                    if self.local_to_global and region_name in self.local_to_global:
                        for organ_local_idx in range(local_logits.size(1)):
                            global_idx = self.local_to_global[region_name][organ_local_idx]
                            organ_logits[batch_idx, global_idx] = local_logits[local_idx, organ_local_idx]
                    else:
                        # No mapping available, use direct indexing
                        num_organs = local_logits.size(1)
                        organ_logits[batch_idx, :num_organs] = local_logits[local_idx]
        
        # Restore original training states
        if isinstance(self.fine_classifier, nn.ModuleDict):
            for region_name, model in self.fine_classifier.items():
                if original_states[region_name]:
                    model.train()
                else:
                    model.eval()
        
        if return_dict:
            return {
                'coarse': {
                    'logits': region_logits,
                    'predictions': region_predictions,
                    'confidence': confidence
                },
                'fine': {
                    'logits': organ_logits
                }
            }
        else:
            # Return only the final organ logits for compatibility with standard loss functions
            return organ_logits
    
    def predict(self, x):
        """
        Make hierarchical predictions.
        
        Returns:
            dict: Region, pathology, and optional subtype predictions
        """
        with torch.no_grad():
            # Stage 1
            region_idx, region_names, region_conf = self.coarse_classifier.predict_region(x)
            
            # Stage 2 - batch processing by region
            batch_size = x.size(0)
            pathology_preds = []
            pathology_conf = []
            
            for i in range(batch_size):
                region_name = region_names[i]
                sample = x[i:i+1]
                
                path_idx, path_conf = self.fine_classifier.get_classifier(
                    region_name
                ).predict_pathology(sample)
                
                pathology_preds.append(path_idx)
                pathology_conf.append(path_conf)
            
            return {
                'region_indices': region_idx,
                'region_names': region_names,
                'region_confidence': region_conf,
                'pathology_indices': torch.cat(pathology_preds),
                'pathology_confidence': torch.cat(pathology_conf)
            }
