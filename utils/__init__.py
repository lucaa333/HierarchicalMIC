"""
Utilities package for hierarchical medical image classification
"""

# Model architectures
from .base_model import Base3DCNN, Enhanced3DCNN, ResidualBlock3D
from .cnn_3d_models import (
    densenet121_3d,
    efficientnet3d_b0,
    get_3d_model,
    resnet18_3d,
    resnet34_3d,
    resnet50_3d,
)
from .coarse_classifier import CoarseAnatomicalClassifier

# Data utilities
from .data_loader import (
    DATASET_TO_REGION,
    REGION_DATASET_MAPPING,
    HierarchicalMedMNISTDataset,
    create_hierarchical_dataset,
    get_medmnist_dataloaders,
)
from .fine_classifier import FinePathologyClassifier, RegionSpecificPathologyNetwork
from .hierarchical_model import HierarchicalClassificationModel

# Evaluation utilities
from .metrics import (
    compute_hierarchical_metrics,
    compute_metrics,
    evaluate_model,
    hierarchical_consistency_score,
)

# Model management utilities
from .model_utils import (
    compare_models,
    count_parameters,
    export_model_to_onnx,
    freeze_layers,
    get_model_size,
    load_checkpoint,
    print_model_summary,
    save_checkpoint,
    unfreeze_all_layers,
)
from .subtype_classifier import HierarchicalSubtypeNetwork, SubtypeClassifier

# Training utilities
from .trainer import HierarchicalTrainer, Trainer

# Visualization utilities
from .visualization import (
    plot_confusion_matrix,
    plot_hierarchical_results,
    plot_metrics_comparison,
    plot_training_history,
    visualize_3d_sample,
)

__all__ = [
    # Models
    "Base3DCNN",
    "Enhanced3DCNN",
    "ResidualBlock3D",
    "CoarseAnatomicalClassifier",
    "MultiScaleCoarseClassifier",
    "FinePathologyClassifier",
    "RegionSpecificPathologyNetwork",
    "AttentionFineClassifier",
    "SubtypeClassifier",
    "HierarchicalSubtypeNetwork",
    "HierarchicalClassificationModel",
    # Data
    "get_medmnist_dataloaders",
    "create_hierarchical_dataset",
    "HierarchicalMedMNISTDataset",
    "REGION_DATASET_MAPPING",
    "DATASET_TO_REGION",
    # Training
    "Trainer",
    "HierarchicalTrainer",
    # Metrics
    "compute_metrics",
    "compute_hierarchical_metrics",
    "evaluate_model",
    "hierarchical_consistency_score",
    # Visualization
    "plot_training_history",
    "visualize_3d_sample",
    "plot_confusion_matrix",
    "plot_hierarchical_results",
    "plot_metrics_comparison",
    # Model utilities
    "save_checkpoint",
    "load_checkpoint",
    "count_parameters",
    "print_model_summary",
    "get_model_size",
    "freeze_layers",
    "unfreeze_all_layers",
    "export_model_to_onnx",
    "compare_models",
]
