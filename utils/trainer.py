"""
Training utilities for hierarchical models
"""

import copy
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
        best_model_state = None
        best_epoch = 0

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

            # Early Stopping Check with Best Model Saving
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
                print(f"New best validation accuracy: {best_val_acc:.4f}")
            else:
                patience_counter += 1
                print(f"Early Stopping Counter: {patience_counter}/{patience}")
                
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered! No improvement for {patience} epochs.")
                break

        # Restore best model weights
        if best_model_state is not None:
            print(f"\nRestoring best model weights from epoch {best_epoch} (val_acc: {best_val_acc:.4f})")
            self.model.load_state_dict(best_model_state)

        return self.history
