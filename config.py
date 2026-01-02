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
# Applied to training and validation sets to reduce overfitting
AUGMENTATION_CONFIG = {
    'enabled': True,
    # Flipping: horizontal/vertical reflections (probability)
    'flip_prob': 0.5,
    # Rotation: counter-clockwise rotation range in degrees
    'rotation_range': (-20, 20),
    # Random cropping: scale and aspect ratio ranges
    'crop_scale': (0.85, 1.0),      # Crop 85-100% of image
    'crop_ratio': (0.9, 1.1),       # Aspect ratio range
    # Shearing: transformation along axes (degrees)
    'shear_range': (-12, 12),
    # Additional augmentations for 3D medical images
    'gaussian_noise_std': 0.02,     # Add slight Gaussian noise
    'brightness_range': (0.9, 1.1), # Brightness adjustment
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
    'architecture': 'efficientnet3d_b0',  # Default: ResNet-18 (3D)
    'coarse_architecture': 'efficientnet3d_b0',  # Architecture for coarse (Stage 1)
    'fine_architecture': 'efficientnet3d_b0',    # Architecture for fine (Stage 2)
    'dropout_rate': 0.3,
    'use_subtypes': False,
}

# Training configuration
TRAINING_CONFIG = {
    'coarse_epochs': 1,
    'fine_epochs': 1,
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
