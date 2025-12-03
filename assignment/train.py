"""
Training Script for Pac-Man Behavior Cloning

This module trains the neural network to imitate human gameplay.
Students will implement the training loop and optimization steps.

Usage:
    python train.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
from pathlib import Path

import config
from model import create_model, save_model
from dataset import create_dataloaders


class Trainer:
    """Handles training and evaluation of the Pac-Man model"""

    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, dataset=None):
        """
        Initialize trainer

        Args:
            model: The neural network to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            dataset: Original dataset (to access normalization stats)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.dataset = dataset

        # ====================================================================
        # TODO 1: Initialize loss function and optimizer
        # ====================================================================
        # Hint: Use nn.CrossEntropyLoss() for multi-class classification
        # Hint: Use optim.Adam() for the optimizer

        # Define loss function (CrossEntropyLoss for classification)
        # Apply class weights if enabled (advanced feature - already implemented)
        if config.USE_CLASS_WEIGHTS and dataset is not None:
            class_weights = dataset.get_class_weights()
            class_weights = class_weights.to(config.DEVICE)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
            print(f"\n[Trainer] Class weighting ENABLED")
            print(f"  Class weights: {class_weights.cpu().numpy()}")
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Define optimizer (Adam with learning rate and weight decay from config)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # History for plotting
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []

        print("\n[Trainer] Initialized")
        print(f"  Optimizer: Adam (lr={config.LEARNING_RATE}, weight_decay={config.WEIGHT_DECAY})")
        loss_type = "CrossEntropyLoss (weighted)" if config.USE_CLASS_WEIGHTS else "CrossEntropyLoss"
        print(f"  Loss function: {loss_type}")
        print(f"  Device: {config.DEVICE}")

    def train_epoch(self) -> tuple:
        """
        Train for one epoch

        This is a KEY method students need to implement!

        Returns:
            avg_loss: Average training loss for the epoch
            avg_acc: Average training accuracy for the epoch
        """
        self.model.train()  # Set model to training mode

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        # ====================================================================
        # TODO 2: Implement training loop
        # ====================================================================
        # Hint: Iterate through batches, compute loss, backpropagate, update weights

        for batch_idx, (states, actions) in enumerate(self.train_loader):
            # Move data to device
            states = states.to(config.DEVICE)
            actions = actions.to(config.DEVICE)

            # Zero the gradients before backward pass
            self.optimizer.zero_grad()

            # Forward pass - get model predictions
            outputs = self.model(states)

            # Compute loss between predictions and true actions
            loss = self.criterion(outputs, actions)

            # Backward pass - compute gradients
            loss.backward()

            # Update weights using computed gradients
            self.optimizer.step()

            # Track statistics (provided)
            total_loss += loss.item() * states.size(0)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == actions).sum().item()
            total_samples += states.size(0)

            # Print progress
            if batch_idx % config.LOG_INTERVAL == 0:
                print(f"  Batch [{batch_idx}/{len(self.train_loader)}] "
                      f"Loss: {loss.item():.4f}")

        # Calculate average metrics
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples

        return avg_loss, avg_acc

    def validate(self) -> tuple:
        """
        Evaluate on validation set

        Returns:
            avg_loss: Average validation loss
            avg_acc: Average validation accuracy
        """
        self.model.eval()  # Set model to evaluation mode

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():  # No gradient computation for validation
            for states, actions in self.val_loader:
                states = states.to(config.DEVICE)
                actions = actions.to(config.DEVICE)

                # Forward pass
                outputs = self.model(states)

                # Compute loss
                loss = self.criterion(outputs, actions)

                # Track statistics
                total_loss += loss.item() * states.size(0)
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == actions).sum().item()
                total_samples += states.size(0)

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples

        return avg_loss, avg_acc

    def train(self, num_epochs: int = config.NUM_EPOCHS):
        """
        Full training loop with early stopping

        Args:
            num_epochs: Number of epochs to train
        """
        print("\n" + "=" * 70)
        print("STARTING TRAINING")
        print("=" * 70)
        print(f"Epochs: {num_epochs}")
        print(f"Batch size: {config.BATCH_SIZE}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print("=" * 70 + "\n")

        start_time = time.time()

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()

            print(f"\nEpoch [{epoch+1}/{num_epochs}]")
            print("-" * 70)

            # Train
            train_loss, train_acc = self.train_epoch()

            # Validate
            val_loss, val_acc = self.validate()

            # Save history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)

            # Print epoch summary
            epoch_time = time.time() - epoch_start
            print("-" * 70)
            print(f"Epoch [{epoch+1}/{num_epochs}] Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
            print(f"  Time: {epoch_time:.2f}s")

            # Save best model
            if val_loss < self.best_val_loss - config.MIN_DELTA:
                self.best_val_loss = val_loss
                self.patience_counter = 0

                if config.SAVE_BEST_MODEL:
                    # Save model with normalization statistics
                    state_mean = self.dataset.state_mean if self.dataset else None
                    state_std = self.dataset.state_std if self.dataset else None
                    save_model(self.model, config.MODEL_SAVE_PATH, state_mean, state_std)
                    print(f"  ✓ New best model saved! (Val Loss: {val_loss:.4f})")
            else:
                self.patience_counter += 1
                print(f"  No improvement ({self.patience_counter}/{config.PATIENCE})")

            # Early stopping
            if self.patience_counter >= config.PATIENCE:
                print(f"\n[Early Stopping] No improvement for {config.PATIENCE} epochs")
                break

        # Training complete
        total_time = time.time() - start_time
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} min)")
        print(f"Best val loss: {self.best_val_loss:.4f}")
        print(f"Final train acc: {self.train_accs[-1]:.4f}")
        print(f"Final val acc: {self.val_accs[-1]:.4f}")
        print("=" * 70 + "\n")

    def get_per_class_accuracy(self) -> dict:
        """
        Compute accuracy for each action class

        Returns:
            class_accs: Dict mapping action names to accuracies
        """
        self.model.eval()

        # Count correct predictions per class
        class_correct = {i: 0 for i in range(config.NUM_ACTIONS)}
        class_total = {i: 0 for i in range(config.NUM_ACTIONS)}

        with torch.no_grad():
            for states, actions in self.val_loader:
                states = states.to(config.DEVICE)
                actions = actions.to(config.DEVICE)

                outputs = self.model(states)
                _, predicted = torch.max(outputs, 1)

                for action, pred in zip(actions, predicted):
                    action = action.item()
                    class_total[action] += 1
                    if pred == action:
                        class_correct[action] += 1

        # Compute accuracies
        class_accs = {}
        for i in range(config.NUM_ACTIONS):
            if class_total[i] > 0:
                acc = class_correct[i] / class_total[i]
                action_name = config.IDX_TO_ACTION[i]
                class_accs[action_name] = acc

        return class_accs

    def print_final_report(self):
        """Print detailed training report"""
        print("\n" + "=" * 70)
        print("TRAINING REPORT")
        print("=" * 70)

        # Per-class accuracy
        class_accs = self.get_per_class_accuracy()
        print("\nPer-Class Accuracy (Validation Set):")
        for action, acc in class_accs.items():
            bar = "█" * int(acc * 50)
            print(f"  {action:8s}: {acc:.4f} ({acc*100:.1f}%) {bar}")

        print("\nTraining History:")
        print(f"  Best Val Loss: {self.best_val_loss:.4f}")
        print(f"  Best Val Acc:  {max(self.val_accs):.4f}")
        print(f"  Final Train Acc: {self.train_accs[-1]:.4f}")
        print(f"  Final Val Acc:   {self.val_accs[-1]:.4f}")

        print("=" * 70 + "\n")


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main():
    """Main training function"""
    print("\n" + "=" * 70)
    print("PAC-MAN BEHAVIOR CLONING - TRAINING")
    print("=" * 70)

    # Print configuration
    config.print_config()

    # Create data loaders
    print("\n[1/3] Loading data...")
    train_loader, val_loader, dataset = create_dataloaders()

    # Create model
    print("\n[2/3] Creating model...")
    model = create_model()

    # Train
    print("\n[3/3] Training model...")
    trainer = Trainer(model, train_loader, val_loader, dataset)
    trainer.train()

    # Print final report
    trainer.print_final_report()

    print("\n✓ Training complete!")
    print(f"Model saved to: {config.MODEL_SAVE_PATH}")
    print("\nNext steps:")
    print("  1. Review the training metrics above")
    print("  2. Run auto_play.py to see your model play Pac-Man!")
    print("  3. Experiment with hyperparameters in config.py to improve performance")


if __name__ == "__main__":
    main()
