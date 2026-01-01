"""
Training utilities for hierarchical models
"""

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


class Trainer:
    """
    Basic trainer for single-stage models.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device=None,
        scheduler=None,
    ):
        from config import DEVICE
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device if device is not None else DEVICE
        self.scheduler = scheduler

        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc="Training")
        for imgs, labels in pbar:
            imgs = imgs.to(self.device, dtype=torch.float32)
            if imgs.max() > 1:
                imgs = imgs / 255.0

            labels = labels.squeeze(-1).long().to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(imgs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)

            pbar.set_postfix({"loss": loss.item(), "acc": correct / total})

        avg_loss = total_loss / total
        avg_acc = correct / total

        return avg_loss, avg_acc

    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, labels in tqdm(self.val_loader, desc="Validation"):
                imgs = imgs.to(self.device, dtype=torch.float32)
                if imgs.max() > 1:
                    imgs = imgs / 255.0

                labels = labels.squeeze(-1).long().to(self.device)

                outputs = self.model(imgs)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * imgs.size(0)
                preds = outputs.argmax(1)
                correct += (preds == labels).sum().item()
                total += imgs.size(0)

        avg_loss = total_loss / total
        avg_acc = correct / total

        return avg_loss, avg_acc

    def train(self, num_epochs):
        """
        Train the model for multiple epochs.
        """
        from config import EXPERIMENT_CONFIG
        patience = EXPERIMENT_CONFIG['early_stopping_patience']
        patience_counter = 0
        best_val_acc = 0.0

        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")

            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            if self.scheduler:
                self.scheduler.step()

            # Early Stopping Check
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                print(f"New best validation accuracy: {best_val_acc:.4f}")
            else:
                patience_counter += 1
                print(f"Early Stopping Counter: {patience_counter}/{patience}")
                
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered! No improvement for {patience} epochs.")
                break

        return self.history


class HierarchicalTrainer:
    """
    Trainer for hierarchical multi-stage models.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device=None,
        coarse_weight=0.3,
        fine_weight=0.7,
    ):
        from config import DEVICE
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device if device is not None else DEVICE
        self.coarse_weight = coarse_weight
        self.fine_weight = fine_weight

        # Separate optimizers for each stage
        self.coarse_optimizer = torch.optim.Adam(
            model.coarse_classifier.parameters(), lr=0.001
        )
        self.fine_optimizers = {}
        for region_name, classifier in model.fine_classifier.classifiers.items():
            self.fine_optimizers[region_name] = torch.optim.Adam(
                classifier.parameters(), lr=0.001
            )

        self.criterion = nn.CrossEntropyLoss()

        self.history = {
            "coarse_train_loss": [],
            "coarse_train_acc": [],
            "fine_train_loss": [],
            "fine_train_acc": [],
            "coarse_val_loss": [],
            "coarse_val_acc": [],
            "fine_val_loss": [],
            "fine_val_acc": [],
        }

    def train_coarse_stage(self, num_epochs):
        """Train the coarse classifier."""
        print("\n=== Training Stage 1: Coarse Anatomical Classifier ===")

        for epoch in range(1, num_epochs + 1):
            self.model.coarse_classifier.train()
            total_loss = 0.0
            correct = 0
            total = 0

            for imgs, coarse_labels, _ in tqdm(
                self.train_loader, desc=f"Epoch {epoch}"
            ):
                imgs = imgs.to(self.device, dtype=torch.float32)
                if imgs.max() > 1:
                    imgs = imgs / 255.0

                coarse_labels = coarse_labels.long().to(self.device)

                self.coarse_optimizer.zero_grad()
                outputs = self.model.forward_coarse(imgs)
                loss = self.criterion(outputs, coarse_labels)
                loss.backward()
                self.coarse_optimizer.step()

                total_loss += loss.item() * imgs.size(0)
                preds = outputs.argmax(1)
                correct += (preds == coarse_labels).sum().item()
                total += imgs.size(0)

            avg_loss = total_loss / total
            avg_acc = correct / total

            print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Acc={avg_acc:.4f}")

            self.history["coarse_train_loss"].append(avg_loss)
            self.history["coarse_train_acc"].append(avg_acc)

    def train_fine_stage(self, num_epochs, freeze_coarse=True):
        """
        Train region-specific fine classifiers.
        
        Stage 1 (coarse classifier) is frozen during this phase by default.
        Set freeze_coarse=False for end-to-end fine-tuning.
        
        Samples are routed to their corresponding region classifier based on
        the coarse label, and each fine classifier is trained on its region's data.
        """
        print("\n=== Training Stage 2: Fine Pathology Classifiers ===")
        
        if freeze_coarse:
            # Freeze Stage 1 (coarse classifier) as per paper methodology
            print("Freezing Stage 1 (coarse classifier)...")
            self.model.coarse_classifier.eval()
            for param in self.model.coarse_classifier.parameters():
                param.requires_grad = False
        else:
            print("End-to-End Training: Stage 1 (coarse classifier) is trainable.")
            self.model.coarse_classifier.train()
            # Ensure parameters are trainable
            for param in self.model.coarse_classifier.parameters():
                param.requires_grad = True
        
        # Get region index to name mapping
        region_idx_to_name = {i: name for i, name in enumerate(self.model.region_configs.keys())}

        for epoch in range(1, num_epochs + 1):
            # Set all fine classifiers to train mode
            for classifier in self.model.fine_classifier.classifiers.values():
                classifier.train()
            
            # Track metrics per region
            region_losses = {name: 0.0 for name in self.model.region_configs.keys()}
            region_correct = {name: 0 for name in self.model.region_configs.keys()}
            region_total = {name: 0 for name in self.model.region_configs.keys()}
            
            for imgs, coarse_labels, fine_labels in tqdm(
                self.train_loader, desc=f"Epoch {epoch}"
            ):
                imgs = imgs.to(self.device, dtype=torch.float32)
                if imgs.max() > 1:
                    imgs = imgs / 255.0
                
                coarse_labels = coarse_labels.long().to(self.device)
                fine_labels = fine_labels.squeeze(-1).long().to(self.device)
                
                # Group samples by region for batch processing
                for region_idx, region_name in region_idx_to_name.items():
                    # Get samples belonging to this region
                    region_mask = (coarse_labels == region_idx)
                    if not region_mask.any():
                        continue
                    
                    region_imgs = imgs[region_mask]
                    region_fine_labels = fine_labels[region_mask]
                    
                    # Zero gradients for this region's optimizer
                    self.fine_optimizers[region_name].zero_grad()
                    
                    # Forward through fine classifier
                    outputs = self.model.forward_fine(region_imgs, region_name)
                    
                    # Compute loss
                    loss = self.criterion(outputs, region_fine_labels)
                    loss.backward()
                    self.fine_optimizers[region_name].step()
                    
                    # Track metrics
                    region_losses[region_name] += loss.item() * region_imgs.size(0)
                    preds = outputs.argmax(1)
                    region_correct[region_name] += (preds == region_fine_labels).sum().item()
                    region_total[region_name] += region_imgs.size(0)
            
            # Compute and display epoch metrics
            total_loss = 0.0
            total_correct = 0
            total_samples = 0
            
            print(f"Epoch {epoch} Results:")
            for region_name in self.model.region_configs.keys():
                if region_total[region_name] > 0:
                    region_avg_loss = region_losses[region_name] / region_total[region_name]
                    region_avg_acc = region_correct[region_name] / region_total[region_name]
                    print(f"  {region_name}: Loss={region_avg_loss:.4f}, Acc={region_avg_acc:.4f}")
                    
                    total_loss += region_losses[region_name]
                    total_correct += region_correct[region_name]
                    total_samples += region_total[region_name]
            
            if total_samples > 0:
                avg_loss = total_loss / total_samples
                avg_acc = total_correct / total_samples
                print(f"  Overall: Loss={avg_loss:.4f}, Acc={avg_acc:.4f}")
                
                self.history["fine_train_loss"].append(avg_loss)
                self.history["fine_train_acc"].append(avg_acc)

    def train(self, coarse_epochs=10, fine_epochs=15):
        """
        Train the hierarchical model in stages.
        """
        # Stage 1: Train coarse classifier
        self.train_coarse_stage(coarse_epochs)

        # Stage 2: Train fine classifiers
        self.train_fine_stage(fine_epochs)

        return self.history
