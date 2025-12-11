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
        # Detect platform based on GPU name
        if 'AMD' in gpu_name or 'Radeon' in gpu_name:
            print("Platform: AMD ROCm")
        else:
            print("Platform: NVIDIA CUDA")
        return device
    else:
        print("No GPU detected, using CPU")
        return torch.device('cpu')

DEVICE = get_device()

# Data configuration
DATA_CONFIG = {
    'batch_size': 8,  # Reduced from 16 to fit GPU memory
    'num_workers': 2,  # Reduced to lower memory overhead
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
    'architecture': 'base',  # Default: ResNet-18 (3D)
    'coarse_architecture': 'base',  # Architecture for coarse (Stage 1)
    'fine_architecture': 'base',    # Architecture for fine (Stage 2)
    'dropout_rate': 0.3,
    'use_subtypes': False,
}

# Region configurations for hierarchical model
# Maps anatomical regions to number of pathology classes
REGION_CONFIGS = {
    'brain': 2,    # e.g., normal vs abnormal or vessel/synapse classes
    'abdomen': 2,  # e.g., adrenal gland classes
    'chest': 2,    # e.g., nodule benign/malignant
}

# Subtype configurations (optional, for 3-stage hierarchy)
SUBTYPE_CONFIGS = {
    'brain': {
        'vessel': 2,
        'synapse': 2,
    },
    'chest': {
        'nodule': 2,
    },
    'abdomen': {
        'adrenal': 2,
    }
}

# Training configuration
TRAINING_CONFIG = {
    'coarse_epochs': 20,
    'fine_epochs': 30,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'scheduler_step_size': 10,
    'scheduler_gamma': 0.5,
}

# OrganMNIST3D class mapping
ORGAN_CLASSES = {
    0: 'bladder',
    1: 'femur-left',
    2: 'femur-right',
    3: 'heart',
    4: 'kidney-left',
    5: 'kidney-right',
    6: 'liver',
    7: 'lung-left',
    8: 'lung-right',
    9: 'pancreas',
    10: 'spleen',
}

# Anatomical region grouping for OrganMNIST3D
ORGAN_TO_REGION = {
    'bladder': 'abdomen',
    'femur-left': 'abdomen',
    'femur-right': 'abdomen',
    'heart': 'chest',
    'kidney-left': 'abdomen',
    'kidney-right': 'abdomen',
    'liver': 'abdomen',
    'lung-left': 'chest',
    'lung-right': 'chest',
    'pancreas': 'abdomen',
    'spleen': 'abdomen',
}

# Dataset selection for experiments
DATASETS_TO_USE = {
    'single_organ': ['organ'],  # Single dataset
    'multi_region': ['nodule', 'adrenal', 'vessel'],  # Multi-region
    'full': ['organ', 'nodule', 'adrenal', 'fracture', 'vessel'],  # All available
}

# Visualization configuration
VIZ_CONFIG = {
    'num_slices': 6,
    'figsize': (15, 5),
    'cmap': 'gray',
    'dpi': 300,
}

# Paths
PATHS = {
    'results': './results/',
    'models': './models/',
    'figures': './figures/',
    'logs': './logs/',
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
