"""
Neural Network Model for Pac-Man Behavior Cloning

This module implements a Multi-Layer Perceptron (MLP) classifier that learns
to predict actions from game states.

Students will implement the network architecture and forward pass.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List

import config


class PacManMLP(nn.Module):
    """
    Multi-Layer Perceptron for Pac-Man action classification

    Architecture:
        Input (STATE_DIM) -> Hidden Layer 1 -> Hidden Layer 2 -> Output (NUM_ACTIONS)

    Students will implement the layers and forward pass.
    """

    def __init__(
        self,
        input_dim: int = config.STATE_DIM,
        hidden_dims: List[int] = config.HIDDEN_DIMS,
        output_dim: int = config.NUM_ACTIONS,
        dropout_rate: float = config.DROPOUT_RATE
    ):
        """
        Initialize the neural network

        Args:
            input_dim: Number of input features (state dimension)
            hidden_dims: List of hidden layer sizes
            output_dim: Number of output classes (actions)
            dropout_rate: Dropout probability for regularization
        """
        super(PacManMLP, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate

        # ====================================================================
        # Define the network layers
        # ====================================================================
        # Architecture: input -> fc1 -> relu -> dropout -> fc2 -> relu -> dropout -> fc3

        # Define first hidden layer (input_dim → hidden_dims[0])
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])

        # Define dropout layer
        self.dropout1 = nn.Dropout(dropout_rate)

        # Define second hidden layer (hidden_dims[0] → hidden_dims[1])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])

        # Define dropout layer
        self.dropout2 = nn.Dropout(dropout_rate)

        # Define output layer (hidden_dims[1] → output_dim)
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)

        # Print architecture summary
        self._print_architecture()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network

        This is a KEY method students need to implement!

        Args:
            x: Input tensor of shape [batch_size, input_dim]

        Returns:
            logits: Output tensor of shape [batch_size, output_dim]
                   (raw scores, not probabilities)
        """

        # ====================================================================
        # Implement the forward pass
        # ====================================================================
        # Architecture: x -> fc1 -> relu -> dropout -> fc2 -> relu -> dropout -> fc3

        # Pass through first layer with ReLU activation
        x = F.relu(self.fc1(x))

        # Apply dropout
        x = self.dropout1(x)

        # Pass through second layer with ReLU activation
        x = F.relu(self.fc2(x))

        # Apply dropout
        x = self.dropout2(x)

        # Pass through output layer (no activation)
        x = self.fc3(x)

        return x

    def predict(self, state: torch.Tensor) -> int:
        """
        Predict action for a single state (for inference/auto-play)

        This method is provided for students.

        Args:
            state: State tensor of shape [STATE_DIM]

        Returns:
            action_idx: Predicted action index (0-4)
        """
        self.eval()  # Set to evaluation mode

        with torch.no_grad():
            # Add batch dimension
            if state.dim() == 1:
                state = state.unsqueeze(0)  # [STATE_DIM] -> [1, STATE_DIM]

            # Forward pass
            logits = self.forward(state)  # [1, NUM_ACTIONS]

            # Get action with highest score
            action_idx = torch.argmax(logits, dim=1).item()

        return action_idx

    def predict_probs(self, state: torch.Tensor) -> torch.Tensor:
        """
        Predict action probabilities for a single state

        Args:
            state: State tensor of shape [STATE_DIM]

        Returns:
            probs: Probability distribution over actions, shape [NUM_ACTIONS]
        """
        self.eval()

        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)

            logits = self.forward(state)
            probs = F.softmax(logits, dim=1).squeeze(0)

        return probs

    def _print_architecture(self):
        """Print model architecture summary"""
        print("\n" + "=" * 70)
        print("PAC-MAN MLP ARCHITECTURE")
        print("=" * 70)
        print(f"Input dimension:    {self.input_dim}")
        print(f"Hidden dimensions:  {self.hidden_dims}")
        print(f"Output dimension:   {self.output_dim}")
        print(f"Dropout rate:       {self.dropout_rate}")
        print("\nLayer structure:")
        print(f"  Input  ->  FC({self.input_dim} -> {self.hidden_dims[0]}) -> ReLU -> Dropout({self.dropout_rate})")
        print(f"  Hidden ->  FC({self.hidden_dims[0]} -> {self.hidden_dims[1]}) -> ReLU -> Dropout({self.dropout_rate})")
        print(f"  Output ->  FC({self.hidden_dims[1]} -> {self.output_dim})")

        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"\nTotal parameters:      {total_params:,}")
        print(f"Trainable parameters:  {trainable_params:,}")
        print("=" * 70 + "\n")


# ============================================================================
# MODEL UTILITIES (Provided for students)
# ============================================================================

def create_model() -> PacManMLP:
    """
    Create and initialize the model

    Returns:
        model: Initialized PacManMLP on configured device
    """
    model = PacManMLP(
        input_dim=config.STATE_DIM,
        hidden_dims=config.HIDDEN_DIMS,
        output_dim=config.NUM_ACTIONS,
        dropout_rate=config.DROPOUT_RATE
    )

    # Move to device (GPU if available)
    model = model.to(config.DEVICE)

    print(f"Model created and moved to device: {config.DEVICE}")

    return model


def save_model(model: PacManMLP, filepath: str = None, state_mean: np.ndarray = None, state_std: np.ndarray = None):
    """
    Save model weights and normalization statistics to disk

    Args:
        model: The model to save
        filepath: Path to save the model (default: config.MODEL_SAVE_PATH)
        state_mean: Mean values for state normalization (optional)
        state_std: Std values for state normalization (optional)
    """
    if filepath is None:
        filepath = config.MODEL_SAVE_PATH

    # Save model state dict and normalization stats
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'input_dim': model.input_dim,
        'hidden_dims': model.hidden_dims,
        'output_dim': model.output_dim,
        'dropout_rate': model.dropout_rate,
    }

    # Add normalization statistics if provided
    if state_mean is not None:
        checkpoint['state_mean'] = state_mean
    if state_std is not None:
        checkpoint['state_std'] = state_std

    torch.save(checkpoint, filepath)

    print(f"[Model] Saved to {filepath}")
    if state_mean is not None and state_std is not None:
        print(f"[Model] Normalization statistics included")


def load_model(filepath: str = None):
    """
    Load model weights and normalization statistics from disk

    Args:
        filepath: Path to load the model from (default: config.MODEL_SAVE_PATH)

    Returns:
        model: Loaded model on configured device
        state_mean: Mean values for state normalization (None if not saved)
        state_std: Std values for state normalization (None if not saved)
    """
    if filepath is None:
        filepath = config.MODEL_SAVE_PATH

    # Load checkpoint
    checkpoint = torch.load(filepath, map_location=config.DEVICE)

    # Create model with saved architecture
    model = PacManMLP(
        input_dim=checkpoint['input_dim'],
        hidden_dims=checkpoint['hidden_dims'],
        output_dim=checkpoint['output_dim'],
        dropout_rate=checkpoint['dropout_rate']
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(config.DEVICE)
    model.eval()  # Set to evaluation mode

    # Load normalization statistics if available
    state_mean = checkpoint.get('state_mean', None)
    state_std = checkpoint.get('state_std', None)

    print(f"[Model] Loaded from {filepath}")
    if state_mean is not None and state_std is not None:
        print(f"[Model] Normalization statistics loaded")
    else:
        print(f"[Model] WARNING: No normalization statistics found in checkpoint")

    return model, state_mean, state_std


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# TESTING / DEMO
# ============================================================================

if __name__ == "__main__":
    print("Testing Model implementation...\n")

    # Create model
    model = create_model()

    # Test forward pass
    print("\n" + "=" * 70)
    print("Testing forward pass...")
    print("=" * 70)

    # Create dummy input
    batch_size = 4
    dummy_input = torch.randn(batch_size, config.STATE_DIM).to(config.DEVICE)

    print(f"Input shape: {dummy_input.shape}")

    try:
        # Forward pass
        output = model(dummy_input)
        print(f"Output shape: {output.shape}")
        print(f"Output logits (first sample): {output[0]}")

        # Test prediction
        print("\n" + "=" * 70)
        print("Testing prediction...")
        print("=" * 70)

        single_state = torch.randn(config.STATE_DIM).to(config.DEVICE)
        action_idx = model.predict(single_state)
        action_name = config.IDX_TO_ACTION[action_idx]

        print(f"Predicted action index: {action_idx}")
        print(f"Predicted action name: {action_name}")

        # Test probability prediction
        probs = model.predict_probs(single_state)
        print(f"\nAction probabilities:")
        for i, prob in enumerate(probs):
            print(f"  {config.IDX_TO_ACTION[i]:8s}: {prob:.4f}")

        print("\n" + "=" * 70)
        print("Model implementation test PASSED!")
        print("=" * 70)

    except Exception as e:
        print(f"\nError during forward pass: {e}")
        print("Make sure you've implemented the TODOs in __init__ and forward")
        import traceback
        traceback.print_exc()
