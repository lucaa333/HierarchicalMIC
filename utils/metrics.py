"""
Evaluation metrics for hierarchical classification
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
import torch


def compute_metrics(y_true, y_pred, labels=None):
    """
    Compute classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Label names (optional)
    
    Returns:
        dict: Dictionary of metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix
    }
    
    # Add per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = \
        precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    
    metrics['per_class'] = {
        'precision': precision_per_class,
        'recall': recall_per_class,
        'f1_score': f1_per_class,
        'support': support_per_class
    }
    
    if labels is not None:
        report = classification_report(y_true, y_pred, target_names=labels, zero_division=0)
        metrics['classification_report'] = report
    
    return metrics


def compute_hierarchical_metrics(
    coarse_true,
    coarse_pred,
    fine_true,
    fine_pred,
    region_names=None
):
    """
    Compute metrics for hierarchical classification.
    
    Args:
        coarse_true: True coarse labels
        coarse_pred: Predicted coarse labels
        fine_true: True fine labels
        fine_pred: Predicted fine labels
        region_names: Names of regions
    
    Returns:
        dict: Hierarchical metrics
    """
    # Coarse-level metrics
    coarse_metrics = compute_metrics(coarse_true, coarse_pred, region_names)
    
    # Fine-level metrics
    fine_metrics = compute_metrics(fine_true, fine_pred)
    
    # Hierarchical accuracy (both levels correct)
    hierarchical_correct = (
        (coarse_true == coarse_pred) & (fine_true == fine_pred)
    ).sum()
    hierarchical_acc = hierarchical_correct / len(coarse_true)
    
    # Consistency check (coarse correct but fine wrong)
    coarse_correct_fine_wrong = (
        (coarse_true == coarse_pred) & (fine_true != fine_pred)
    ).sum()
    
    results = {
        'coarse_metrics': coarse_metrics,
        'fine_metrics': fine_metrics,
        'hierarchical_accuracy': hierarchical_acc,
        'coarse_correct_fine_wrong': coarse_correct_fine_wrong
    }
    
    return results


def evaluate_model(model, data_loader, device='cuda'):
    """
    Evaluate a model on a dataset.
    
    Args:
        model: Model to evaluate
        data_loader: DataLoader with test data
        device: Device to use
    
    Returns:
        dict: Evaluation results
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, labels in data_loader:
            imgs = imgs.to(device, dtype=torch.float32)
            if imgs.max() > 1:
                imgs = imgs / 255.0
            
            outputs = model(imgs)
            preds = outputs.argmax(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.squeeze(-1).numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    metrics = compute_metrics(all_labels, all_preds)
    
    return metrics, all_preds, all_labels


def hierarchical_consistency_score(coarse_pred, fine_pred, hierarchy_map):
    """
    Check consistency between hierarchical levels.
    
    Args:
        coarse_pred: Predicted coarse labels
        fine_pred: Predicted fine labels
        hierarchy_map: Mapping from fine to coarse labels
    
    Returns:
        float: Consistency score
    """
    expected_coarse = np.array([hierarchy_map.get(f, -1) for f in fine_pred])
    consistent = (coarse_pred == expected_coarse).sum()
    return consistent / len(coarse_pred)