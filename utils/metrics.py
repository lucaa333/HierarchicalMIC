"""
Evaluation metrics for hierarchical classification
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize
import torch


def compute_auc_metrics(model, data_loader, device, num_classes, task_name="classification"):
    """
    Compute AUC metrics for multi-class classification.
    Returns both macro and weighted AUC scores.
    Handles binary classification (2 classes) correctly.
    
    Args:
        model: PyTorch model to evaluate
        data_loader: DataLoader with evaluation data
        device: Device to run inference on
        num_classes: Number of classes
        task_name: Name of the task (for logging)
    
    Returns:
        dict: Dictionary containing AUC metrics
    """
    model.eval()
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for imgs, labels in data_loader:
            imgs = imgs.to(device, dtype=torch.float32)
            if imgs.max() > 1:
                imgs = imgs / 255.0
            
            labels_flat = labels.view(-1)
            
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)
            
            all_labels.extend(labels_flat.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Handle binary vs multi-class case
    # For binary classification, label_binarize returns shape (n_samples, 1)
    # We need to handle this specially
    if num_classes == 2:
        # Binary classification: use probability of positive class directly
        try:
            auc_macro = roc_auc_score(all_labels, all_probs[:, 1])
            auc_weighted = auc_macro  # Same for binary
        except ValueError as e:
            print(f"Warning: Could not compute AUC - {e}")
            auc_macro = 0.0
            auc_weighted = 0.0
        
        # Per-class AUC (same value for both classes in binary)
        per_class_auc = [auc_macro, auc_macro]
    else:
        # Multi-class: binarize labels
        labels_binarized = label_binarize(all_labels, classes=range(num_classes))
        
        # Compute AUC scores
        try:
            auc_macro = roc_auc_score(labels_binarized, all_probs, average='macro', multi_class='ovr')
            auc_weighted = roc_auc_score(labels_binarized, all_probs, average='weighted', multi_class='ovr')
        except ValueError as e:
            print(f"Warning: Could not compute AUC - {e}")
            auc_macro = 0.0
            auc_weighted = 0.0
        
        # Compute per-class AUC
        per_class_auc = []
        for i in range(num_classes):
            if len(np.unique(labels_binarized[:, i])) > 1:
                try:
                    class_auc = roc_auc_score(labels_binarized[:, i], all_probs[:, i])
                    per_class_auc.append(class_auc)
                except:
                    per_class_auc.append(0.0)
            else:
                per_class_auc.append(0.0)
    
    return {
        'auc_macro': auc_macro,
        'auc_weighted': auc_weighted,
        'per_class_auc': per_class_auc,
        'all_labels': all_labels,
        'all_probs': all_probs
    }


def compute_auc(y_true, y_prob, multi_class='ovr'):
    """
    Compute AUC (Area Under the ROC Curve) for classification.
    
    Args:
        y_true: True labels (integers)
        y_prob: Predicted probabilities, shape (n_samples, n_classes)
        multi_class: Strategy for multi-class AUC ('ovr' or 'ovo')
    
    Returns:
        float: AUC score (or None if computation fails)
    """
    try:
        y_true = np.array(y_true)
        y_prob = np.array(y_prob)
        
        # For multi-class, use One-vs-Rest approach
        n_classes = y_prob.shape[1] if len(y_prob.shape) > 1 else 2
        
        if n_classes == 2:
            # Binary classification
            probs = y_prob[:, 1] if len(y_prob.shape) > 1 else y_prob
            return roc_auc_score(y_true, probs)
        else:
            # Multi-class classification
            return roc_auc_score(y_true, y_prob, multi_class=multi_class)
    except Exception as e:
        print(f"Warning: Could not compute AUC: {e}")
        return None


def compute_metrics(y_true, y_pred, labels=None, y_prob=None):
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


def evaluate_model(model, data_loader, device=None):
    """
    Evaluate a model on a dataset.
    
    Args:
        model: Model to evaluate
        data_loader: DataLoader with test data
        device: Device to use (defaults to config.DEVICE)
    
    Returns:
        dict: Evaluation results
    """
    from config import DEVICE
    if device is None:
        device = DEVICE
    
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


def save_metrics_json(metrics_dict, filepath, indent=2):
    """
    Save metrics to JSON file with timestamp.
    
    Args:
        metrics_dict: Dictionary containing metrics
        filepath: Path to save JSON file
        indent: JSON indentation (default: 2)
    """
    import json
    import os
    from datetime import datetime
    
    # Add timestamp if not present
    if 'timestamp' not in metrics_dict:
        metrics_dict['timestamp'] = datetime.now().isoformat()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save
    with open(filepath, 'w') as f:
        json.dump(metrics_dict, f, indent=indent)
    
    print(f"Metrics saved to: {filepath}")


def compute_comprehensive_metrics(y_true, y_pred, y_prob=None, class_names=None):
    """
    Compute all standard metrics in the schema format.
    
    Args:
        y_true: True labels (array-like)
        y_pred: Predicted labels (array-like)
        y_prob: Predicted probabilities (optional, for AUC) - shape (n_samples, n_classes)
        class_names: List of class names (optional)
    
    Returns:
        dict: Metrics in standard schema format
    """
    from sklearn.preprocessing import label_binarize
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    # Per-class metrics
    prec_per_class, rec_per_class, f1_per_class, support_per_class = \
        precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    
    # AUC (if probabilities provided)
    auc = None
    if y_prob is not None:
        try:
            y_prob = np.array(y_prob)
            num_classes = y_prob.shape[1] if len(y_prob.shape) > 1 else 2
            
            if num_classes == 2:
                # Binary classification
                auc = roc_auc_score(y_true, y_prob[:, 1])
            else:
                # Multi-class classification
                y_true_bin = label_binarize(y_true, classes=range(num_classes))
                auc = roc_auc_score(y_true_bin, y_prob, average='weighted', multi_class='ovr')
        except Exception as e:
            print(f"Warning: Could not compute AUC - {e}")
            auc = None
    
    # Build metrics dict
    metrics = {
        "accuracy": float(accuracy),
        "precision_weighted": float(prec),
        "recall_weighted": float(rec),
        "f1_weighted": float(f1),
    }
    
    if auc is not None:
        metrics["auc_weighted"] = float(auc)
    
    metrics["per_class"] = {
        "precision": [float(x) for x in prec_per_class],
        "recall": [float(x) for x in rec_per_class],
        "f1": [float(x) for x in f1_per_class],
        "support": [int(x) for x in support_per_class]
    }
    
    if class_names is not None:
        metrics["class_names"] = class_names
    
    return metrics