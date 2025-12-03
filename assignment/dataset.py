"""
PyTorch Dataset for Pac-Man Behavior Cloning

This module implements a PyTorch Dataset that loads recorded trajectories
and prepares them for training the neural network.

Students will implement key methods for data loading and preprocessing.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import List, Tuple, Dict
from pathlib import Path

import config
from data_recorder import load_all_trajectories


class PacManDataset(Dataset):
    """PyTorch Dataset for behavior cloning from recorded trajectories"""

    def __init__(self, trajectories: List[Dict], normalize: bool = True):
        """
        Initialize dataset from list of trajectories

        Args:
            trajectories: List of dicts with "states" and "actions" arrays
            normalize: Whether to normalize state features (recommended)
        """
        self.normalize = normalize

        # Combine all trajectories into single arrays
        all_states = []
        all_actions = []

        for traj in trajectories:
            all_states.append(traj["states"])
            all_actions.append(traj["actions"])

        # Concatenate into single arrays
        self.states = np.concatenate(all_states, axis=0)  # Shape: [N, STATE_DIM]
        self.actions = np.concatenate(all_actions, axis=0)  # Shape: [N]

        print(f"[Dataset] Loaded {len(trajectories)} trajectories")
        print(f"  Total samples: {len(self.states):,}")
        print(f"  State shape: {self.states.shape}")
        print(f"  Actions shape: {self.actions.shape}")

        # Compute normalization statistics
        if self.normalize:
            self._compute_normalization_stats()

    def _compute_normalization_stats(self):
        """
        Compute mean and std for state normalization

        This is provided for students - standardization helps neural network training
        """
        self.state_mean = np.mean(self.states, axis=0)
        self.state_std = np.std(self.states, axis=0) + 1e-8  # Add epsilon to avoid division by zero

        print(f"[Dataset] Computed normalization statistics")
        print(f"  State mean: {self.state_mean[:5]}... (showing first 5)")
        print(f"  State std:  {self.state_std[:5]}... (showing first 5)")

    def __len__(self) -> int:
        """Return the number of samples in the dataset"""
        return len(self.states)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single (state, action) pair

        This is a KEY method students need to implement!

        Args:
            idx: Index of the sample to retrieve

        Returns:
            state: Tensor of shape [STATE_DIM]
            action: Tensor (scalar) with action index
        """

        # ====================================================================
        # Get state and action at index idx
        # ====================================================================

        # Get the state vector and action at this index
        state = self.states[idx]
        action = self.actions[idx]


        # ====================================================================
        # Normalize the state if self.normalize is True
        # ====================================================================
        # Standardization formula: (x - mean) / std

        if self.normalize:
            # Normalize state using standardization formula
            state = (state - self.state_mean) / self.state_std


        # ====================================================================
        # FINAL: Convert to PyTorch tensors (provided)
        # ====================================================================
        # Students don't need to modify this part

        state_tensor = torch.from_numpy(state).float()
        action_tensor = torch.tensor(action, dtype=torch.long)

        return state_tensor, action_tensor

    def get_action_distribution(self) -> Dict[str, int]:
        """Get distribution of actions in the dataset"""
        unique, counts = np.unique(self.actions, return_counts=True)

        distribution = {}
        for idx, count in zip(unique, counts):
            action_name = config.IDX_TO_ACTION[int(idx)]
            distribution[action_name] = int(count)

        return distribution

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute class weights for balanced training (inverse frequency)

        Useful if action distribution is imbalanced
        """
        unique, counts = np.unique(self.actions, return_counts=True)

        # Compute inverse frequency weights
        weights = 1.0 / counts
        weights = weights / weights.sum()  # Normalize

        # Create weight tensor for all classes
        class_weights = torch.zeros(config.NUM_ACTIONS)
        for idx, weight in zip(unique, weights):
            class_weights[int(idx)] = weight

        return class_weights

    def balance_dataset(self):
        """
        Balance dataset by oversampling minority classes

        This method duplicates samples from underrepresented actions
        to match the count of the most common action.
        """
        unique, counts = np.unique(self.actions, return_counts=True)
        max_count = counts.max()

        print(f"\n[Dataset] Balancing dataset...")
        print(f"  Before balancing: {len(self.states):,} samples")

        # Store original data
        balanced_states = []
        balanced_actions = []

        # For each action class
        for action_idx, count in zip(unique, counts):
            # Get indices of this action
            action_mask = (self.actions == action_idx)
            action_states = self.states[action_mask]
            action_actions = self.actions[action_mask]

            # Add original samples
            balanced_states.append(action_states)
            balanced_actions.append(action_actions)

            # Oversample to match max_count
            if count < max_count:
                n_to_add = max_count - count
                # Randomly sample with replacement
                oversample_indices = np.random.choice(len(action_states), size=n_to_add, replace=True)
                balanced_states.append(action_states[oversample_indices])
                balanced_actions.append(action_actions[oversample_indices])

                action_name = config.IDX_TO_ACTION[int(action_idx)]
                print(f"    {action_name:8s}: {count:6d} -> {max_count:6d} (+{n_to_add:6d} oversampled)")

        # Concatenate all balanced data
        self.states = np.concatenate(balanced_states, axis=0)
        self.actions = np.concatenate(balanced_actions, axis=0)

        # Shuffle the dataset
        shuffle_indices = np.random.permutation(len(self.states))
        self.states = self.states[shuffle_indices]
        self.actions = self.actions[shuffle_indices]

        print(f"  After balancing: {len(self.states):,} samples")
        print(f"  All actions now have {max_count:,} samples each\n")

        # Recompute normalization statistics with balanced data
        if self.normalize:
            self._compute_normalization_stats()


# ============================================================================
# DATA LOADING UTILITIES (Provided for students)
# ============================================================================

def create_dataloaders(
    train_split: float = config.TRAIN_SPLIT,
    batch_size: int = config.BATCH_SIZE,
    shuffle: bool = True
) -> Tuple[DataLoader, DataLoader, 'PacManDataset']:
    """
    Create train and validation DataLoaders from saved trajectories

    Args:
        train_split: Fraction of data to use for training (rest for validation)
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle training data

    Returns:
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
        dataset: Original dataset (contains normalization statistics)
    """

    # Load all trajectories
    trajectories = load_all_trajectories()

    if len(trajectories) == 0:
        raise ValueError("No trajectories found! Record some gameplay first.")

    # Create dataset
    dataset = PacManDataset(trajectories, normalize=True)

    # Balance dataset if enabled
    if config.BALANCE_DATASET:
        dataset.balance_dataset()

    # Split into train/val
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Reproducible split
    )

    print(f"\n[DataLoaders] Created train/val split")
    print(f"  Train size: {train_size:,} samples")
    print(f"  Val size:   {val_size:,} samples")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Set to 0 for compatibility
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return train_loader, val_loader, dataset


def analyze_dataset():
    """
    Analyze the dataset and print statistics

    Useful for understanding the data before training
    """
    trajectories = load_all_trajectories()

    if len(trajectories) == 0:
        print("No trajectories found!")
        return

    dataset = PacManDataset(trajectories, normalize=False)

    print("\n" + "=" * 70)
    print("DATASET ANALYSIS")
    print("=" * 70)

    # Action distribution
    action_dist = dataset.get_action_distribution()
    print("\nAction Distribution:")
    total_actions = sum(action_dist.values())
    for action, count in action_dist.items():
        percentage = (count / total_actions) * 100
        bar = "█" * int(percentage / 2)
        print(f"  {action:8s}: {count:6d} ({percentage:5.2f}%) {bar}")

    # State statistics
    print(f"\nState Statistics:")
    print(f"  Min values: {dataset.states.min(axis=0)[:5]}... (first 5 features)")
    print(f"  Max values: {dataset.states.max(axis=0)[:5]}... (first 5 features)")
    print(f"  Mean values: {dataset.states.mean(axis=0)[:5]}... (first 5 features)")
    print(f"  Std values: {dataset.states.std(axis=0)[:5]}... (first 5 features)")

    # Class balance
    if config.USE_CLASS_WEIGHTS:
        weights = dataset.get_class_weights()
        print(f"\nClass Weights (for balanced training):")
        for i, weight in enumerate(weights):
            if weight > 0:
                action_name = config.IDX_TO_ACTION[i]
                print(f"  {action_name:8s}: {weight:.4f}")

    print("=" * 70)


# ============================================================================
# TESTING / DEMO
# ============================================================================

if __name__ == "__main__":
    print("Testing Dataset implementation...\n")

    # Check if trajectories exist
    trajectory_files = list(config.TRAJECTORY_DIR.glob("*.pkl"))

    if len(trajectory_files) == 0:
        print("=" * 70)
        print("WARNING: No trajectories found!")
        print("Please record some gameplay first using data_recorder.py")
        print("=" * 70)
    else:
        # Analyze dataset
        analyze_dataset()

        # Test DataLoader creation
        print("\n" + "=" * 70)
        print("Testing DataLoader creation...")
        print("=" * 70)

        try:
            train_loader, val_loader = create_dataloaders(batch_size=32)

            # Test getting a batch
            states, actions = next(iter(train_loader))
            print(f"\nSample batch:")
            print(f"  States shape: {states.shape}")
            print(f"  Actions shape: {actions.shape}")
            print(f"  States dtype: {states.dtype}")
            print(f"  Actions dtype: {actions.dtype}")

            print("\n" + "=" * 70)
            print("Dataset implementation test PASSED!")
            print("=" * 70)

        except Exception as e:
            print(f"\nError creating DataLoaders: {e}")
            print("Make sure you've implemented the TODOs in __getitem__")
