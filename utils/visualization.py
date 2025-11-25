"""
Visualization utilities for medical images and results
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns


def plot_training_history(history, save_path=None):
    """
    Plot training and validation metrics.
    
    Args:
        history (dict): Training history
        save_path (str): Path to save figure (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    if 'train_loss' in history:
        axes[0].plot(history['train_loss'], label='Train Loss')
        axes[0].plot(history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    if 'train_acc' in history:
        axes[1].plot(history['train_acc'], label='Train Acc')
        axes[1].plot(history['val_acc'], label='Val Acc')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def visualize_3d_sample(volume, label=None, num_slices=6, cmap='gray'):
    """
    Visualize slices from a 3D medical image volume.
    
    Args:
        volume: 3D numpy array or torch tensor
        label: Label for the sample (optional)
        num_slices: Number of slices to display
        cmap: Colormap to use
    """
    if isinstance(volume, torch.Tensor):
        volume = volume.cpu().numpy()
    
    if volume.ndim == 4:  # (C, D, H, W)
        volume = volume[0]  # Take first channel
    
    depth = volume.shape[0]
    slice_indices = np.linspace(0, depth-1, num_slices, dtype=int)
    
    fig, axes = plt.subplots(1, num_slices, figsize=(15, 3))
    
    for idx, slice_idx in enumerate(slice_indices):
        axes[idx].imshow(volume[slice_idx], cmap=cmap)
        axes[idx].set_title(f'Slice {slice_idx}')
        axes[idx].axis('off')
    
    if label is not None:
        fig.suptitle(f'Label: {label}', fontsize=14)
    
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(conf_matrix, class_names=None, save_path=None, title='Confusion Matrix'):
    """
    Plot confusion matrix.
    
    Args:
        conf_matrix: Confusion matrix
        class_names: Names of classes
        save_path: Path to save figure
        title: Title for the plot
    """
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names if class_names else 'auto',
        yticklabels=class_names if class_names else 'auto'
    )
    
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_hierarchical_results(coarse_metrics, fine_metrics, save_path=None):
    """
    Plot hierarchical classification results.
    
    Args:
        coarse_metrics: Coarse-level metrics
        fine_metrics: Fine-level metrics
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Coarse-level confusion matrix
    sns.heatmap(
        coarse_metrics['confusion_matrix'],
        annot=True,
        fmt='d',
        cmap='Blues',
        ax=axes[0]
    )
    axes[0].set_title(f"Stage 1: Coarse Classification\nAcc: {coarse_metrics['accuracy']:.3f}")
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    
    # Fine-level confusion matrix
    sns.heatmap(
        fine_metrics['confusion_matrix'],
        annot=True,
        fmt='d',
        cmap='Greens',
        ax=axes[1]
    )
    axes[1].set_title(f"Stage 2: Fine Classification\nAcc: {fine_metrics['accuracy']:.3f}")
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_metrics_comparison(flat_metrics, hierarchical_metrics, save_path=None):
    """
    Compare flat vs hierarchical model performance.
    
    Args:
        flat_metrics: Metrics from flat classifier
        hierarchical_metrics: Metrics from hierarchical classifier
        save_path: Path to save figure
    """
    metrics_names = ['accuracy', 'precision', 'recall', 'f1_score']
    flat_values = [flat_metrics[m] for m in metrics_names]
    hier_values = [hierarchical_metrics['fine_metrics'][m] for m in metrics_names]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, flat_values, width, label='Flat Model', alpha=0.8)
    bars2 = ax.bar(x + width/2, hier_values, width, label='Hierarchical Model', alpha=0.8)
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title('Flat vs Hierarchical Model Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics_names])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.0])
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()