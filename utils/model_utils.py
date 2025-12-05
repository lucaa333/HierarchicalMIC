"""
Model utility functions for saving, loading, and analysis
"""

import torch
import os
from pathlib import Path


def save_checkpoint(model, optimizer, epoch, metrics, filepath):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        metrics: Dictionary of metrics
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(model, optimizer, filepath, device=None):
    """
    Load model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        filepath: Path to checkpoint
        device: Device to load to (defaults to config.DEVICE)
    
    Returns:
        tuple: (epoch, metrics)
    """
    from config import DEVICE
    if device is None:
        device = DEVICE
    
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    metrics = checkpoint.get('metrics', {})
    
    print(f"Checkpoint loaded from {filepath}")
    print(f"Resuming from epoch {epoch}")
    
    return epoch, metrics


def count_parameters(model):
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
    
    Returns:
        tuple: (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params


def print_model_summary(model):
    """
    Print model summary with layer information.
    
    Args:
        model: PyTorch model
    """
    print("\n" + "="*70)
    print("MODEL SUMMARY")
    print("="*70)
    
    total_params, trainable_params = count_parameters(model)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    print("\nModel Architecture:")
    print("-"*70)
    print(model)
    print("="*70 + "\n")


def get_model_size(model):
    """
    Calculate model size in MB.
    
    Args:
        model: PyTorch model
    
    Returns:
        float: Model size in MB
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb


def freeze_layers(model, freeze_until_layer=None):
    """
    Freeze model layers for transfer learning.
    
    Args:
        model: PyTorch model
        freeze_until_layer: Layer name to freeze until (None = freeze all)
    """
    freeze_all = freeze_until_layer is None
    
    for name, param in model.named_parameters():
        if freeze_all:
            param.requires_grad = False
        else:
            if freeze_until_layer in name:
                break
            param.requires_grad = False
    
    trainable, total = count_parameters(model)
    print(f"Frozen layers. Trainable: {trainable:,} / Total: {total:,}")


def unfreeze_all_layers(model):
    """
    Unfreeze all model layers.
    
    Args:
        model: PyTorch model
    """
    for param in model.parameters():
        param.requires_grad = True
    
    print("All layers unfrozen")


def export_model_to_onnx(model, filepath, input_shape=(1, 1, 28, 28, 28)):
    """
    Export PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model
        filepath: Output filepath
        input_shape: Input tensor shape
    """
    model.eval()
    dummy_input = torch.randn(input_shape)
    
    torch.onnx.export(
        model,
        dummy_input,
        filepath,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Model exported to ONNX: {filepath}")


def compare_models(model1, model2):
    """
    Compare two models' architectures and parameters.
    
    Args:
        model1: First PyTorch model
        model2: Second PyTorch model
    """
    params1_total, params1_trainable = count_parameters(model1)
    params2_total, params2_trainable = count_parameters(model2)
    
    size1 = get_model_size(model1)
    size2 = get_model_size(model2)
    
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    print(f"\nModel 1:")
    print(f"  Total parameters: {params1_total:,}")
    print(f"  Trainable parameters: {params1_trainable:,}")
    print(f"  Size: {size1:.2f} MB")
    
    print(f"\nModel 2:")
    print(f"  Total parameters: {params2_total:,}")
    print(f"  Trainable parameters: {params2_trainable:,}")
    print(f"  Size: {size2:.2f} MB")
    
    print(f"\nDifference:")
    print(f"  Parameters: {params2_total - params1_total:+,}")
    print(f"  Size: {size2 - size1:+.2f} MB")
    print("="*70 + "\n")