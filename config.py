"""
Configuration for hierarchical medical image classification
"""

import torch


# Device configuration
def get_device():
    """
    Detect and return the best available device (CUDA/ROCm GPU or CPU).
    Works for both NVIDIA (CUDA) and AMD (ROCm) GPUs.
    
    Note: PyTorch-ROCm uses the same torch.cuda API as NVIDIA CUDA,
    so torch.cuda.is_available() returns True for both GPU types.
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU detected: {gpu_name}")
        print(f"GPU memory: {gpu_memory:.2f} GB")
        return device
    else:
        print("No GPU detected, using CPU")
        return torch.device('cpu')

DEVICE = get_device()

# Data configuration
DATA_CONFIG = {
    'batch_size': 16,  # Reduced from 16 to fit GPU memory
    'num_workers': 8,  # Reduced to lower memory overhead
    'download': True,
}

# Data augmentation configuration
# Applied to training set only to reduce overfitting
AUGMENTATION_CONFIG = {
    'enabled': True,
    # Geometry (conservative defaults for 28^3 volumes)
    'flip_prob': 0.2,
    'flip_axes': (0, 1),  # Avoid width axis (left/right laterality)
    'rotation_prob': 0.5,
    'rotation_range': (-10, 10),
    'rotation_axes': (1, 2),  # Axial plane
    'scale_prob': 0.4,
    'scale_range': (0.95, 1.05),
    'scale_axes': (0, 1, 2),
    'translation_prob': 0.3,
    'translation_range': (-2, 2),
    'translation_axes': (1, 2),
    'crop_prob': 0.0,
    'crop_scale': (0.9, 1.0),
    'crop_ratio': (0.95, 1.05),
    'shear_prob': 0.0,
    'shear_range': (-5, 5),
    # Intensity
    'gaussian_noise_prob': 0.2,
    'gaussian_noise_std': 0.01,
    'brightness_prob': 0.2,
    'brightness_range': (0.95, 1.05),
    'contrast_prob': 0.2,
    'contrast_range': (0.95, 1.05),
    'gamma_prob': 0.1,
    'gamma_range': (0.9, 1.1),
    # Final clip range
    'clip_min': 0.0,
    'clip_max': 1.0,
}

# Model architecture configuration
MODEL_CONFIG = {
    # Available architectures:
    # - 'base': Lightweight Base3DCNN (~0.9M params)
    # - 'enhanced': Enhanced3DCNN with residual blocks
    # - 'resnet18_3d': ResNet-18 (3D) - Recommended default (~33M params)
    # - 'resnet34_3d': ResNet-34 (3D) - More capacity (~63M params)
    # - 'resnet50_3d': ResNet-50 (3D) - Maximum performance (~46M params)
    # - 'densenet121_3d': DenseNet-121 (3D) - Best for limited data (~5.6M params)
    # - 'efficientnet3d_b0': EfficientNet-B0 (3D) - Most efficient (~1.2M params)
    'architecture': 'densenet121_3d',  # Default: ResNet-18 (3D)
    'coarse_architecture': 'densenet121_3d',  # Architecture for coarse (Stage 1)
    'fine_architecture': 'densenet121_3d',    # Architecture for fine (Stage 2)
    'dropout_rate': 0.3,
    'use_subtypes': False,
}

# Training configuration
TRAINING_CONFIG = {
    'coarse_epochs': 20,
    'fine_epochs': 20,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'scheduler_step_size': 10,
    'scheduler_gamma': 0.5,
}


# Dataset selection for experiments
DATASETS_TO_USE = {
    'single_organ': ['organ'],  # Single dataset
    'multi_region': ['nodule', 'adrenal', 'vessel'],  # Multi-region
    'full': ['organ', 'nodule', 'adrenal', 'fracture', 'vessel'],  # All available
}

# Default: All 5 3D MedMNIST datasets for hierarchical training
DEFAULT_MERGED_DATASETS = ['organ', 'nodule', 'adrenal', 'fracture', 'vessel']

# Mapping from short dataset names to MedMNIST info keys
DATASET_INFO_KEYS = {
    'organ': 'organmnist3d',
    'nodule': 'nodulemnist3d',
    'adrenal': 'adrenalmnist3d',
    'fracture': 'fracturemnist3d',
    'vessel': 'vesselmnist3d',
}

# Visualization configuration
VIZ_CONFIG = {
    'num_slices': 6,
    'figsize': (15, 5),
    'cmap': 'gray',
    'dpi': 300,
}

# Paths - use absolute paths based on project root
import os as _os
_PROJECT_ROOT = _os.path.dirname(_os.path.abspath(__file__))

PATHS = {
    'results': _os.path.join(_PROJECT_ROOT, 'results'),
    'models': _os.path.join(_PROJECT_ROOT, 'models'),
    'figures': _os.path.join(_PROJECT_ROOT, 'figures'),
    'logs': _os.path.join(_PROJECT_ROOT, 'logs'),
}

# Experiment settings
EXPERIMENT_CONFIG = {
    'name': 'hierarchical_medmnist3d',
    'seed': 42,
    'save_freq': 5,  # Save model every N epochs
    'early_stopping_patience': 10,
}

# Set random seeds for reproducibility
def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Initialize seed
set_seed(EXPERIMENT_CONFIG['seed'])
