"""
Auto-Play Mode for Pac-Man Behavior Cloning

This module enables the trained neural network to play Pac-Man autonomously.
Students will implement the inference logic to predict actions from game states.

Usage:
    python auto_play.py

The game will run with the model controlling Pac-Man instead of keyboard input.
"""

import torch
import numpy as np
from pathlib import Path
import sys

import config
from model import load_model
from data_recorder import extract_state_features


class AIPlayer:
    """AI player that uses trained model to play Pac-Man"""

    def __init__(self, model_path: Path = None):
        """
        Initialize AI player

        Args:
            model_path: Path to trained model weights
        """
        if model_path is None:
            model_path = config.MODEL_SAVE_PATH

        # ====================================================================
        # Load the trained model
        # ====================================================================

        # Load the trained model (returns model, state_mean, state_std)
        self.model, self.state_mean, self.state_std = load_model(model_path)

        # Fallback: If normalization stats not in checkpoint, compute from trajectories
        if self.state_mean is None or self.state_std is None:
            print("[AI Player] WARNING: Computing normalization stats from trajectories...")
            from data_recorder import load_all_trajectories
            from dataset import PacManDataset
            trajectories = load_all_trajectories()
            if trajectories:
                temp_dataset = PacManDataset(trajectories, normalize=True)
                self.state_mean = temp_dataset.state_mean
                self.state_std = temp_dataset.state_std
                print("[AI Player] Normalization stats computed successfully")
            else:
                print("[AI Player] ERROR: No trajectories found! Cannot normalize inputs!")
                self.state_mean = None
                self.state_std = None

        # Statistics
        self.action_counts = {action: 0 for action in config.ACTION_TO_IDX.keys()}
        self.total_predictions = 0
        self.confidence_scores = []

        # Sliding window for prediction smoothing (majority voting)
        self.use_sliding_window = config.USE_SLIDING_WINDOW
        self.window_size = config.SLIDING_WINDOW_SIZE
        self.prediction_buffer = []  # Store recent raw predictions

        print("\n[AI Player] Initialized")
        print(f"  Model loaded from: {model_path}")
        print(f"  Device: {config.DEVICE}")
        if self.state_mean is not None:
            print(f"  Normalization: ENABLED ✓")
        if self.use_sliding_window:
            print(f"  Prediction smoothing: ENABLED (window size: {self.window_size})")

    def get_action(self, game_state: dict, return_probs: bool = False) -> int:
        """
        Predict action from current game state

        This is a KEY method students need to implement!

        Args:
            game_state: Dictionary with current game state
            return_probs: If True, also return action probabilities

        Returns:
            action_idx: Predicted action index (0-4)
            probs: (Optional) Action probabilities if return_probs=True
        """

        # ====================================================================
        # Extract state features from game state
        # ====================================================================

        # Extract state features from game state
        state_vector = extract_state_features(game_state)


        # ====================================================================
        # Convert state to PyTorch tensor and normalize
        # ====================================================================

        # CRITICAL: Normalize state using training statistics
        if self.state_mean is not None and self.state_std is not None:
            state_vector = (state_vector - self.state_mean) / self.state_std

        # Convert to tensor
        state_tensor = torch.from_numpy(state_vector).float()

        # Move to device
        state_tensor = state_tensor.to(config.DEVICE)


        # ====================================================================
        # Get model prediction
        # ====================================================================

        # Predict action using the model (raw prediction)
        raw_action_idx = self.model.predict(state_tensor)

        # ====================================================================
        # Sliding window prediction smoothing (majority voting)
        # ====================================================================
        if self.use_sliding_window:
            # Add raw prediction to buffer
            self.prediction_buffer.append(raw_action_idx)

            # Keep buffer size limited to window size
            if len(self.prediction_buffer) > self.window_size:
                self.prediction_buffer.pop(0)

            # Use majority voting to get final action
            # Count occurrences of each action in the buffer
            from collections import Counter
            vote_counts = Counter(self.prediction_buffer)

            # Get most common action (majority vote)
            # In case of tie, most_common()[0] returns the first one encountered
            action_idx = vote_counts.most_common(1)[0][0]

            # DEBUG: Show voting details
            if config.DEBUG_MODE and len(self.prediction_buffer) >= 3:
                print(f"\n[SLIDING WINDOW] Buffer: {self.prediction_buffer}")
                print(f"  Raw prediction: {config.IDX_TO_ACTION[raw_action_idx]}")
                print(f"  Majority vote: {config.IDX_TO_ACTION[action_idx]}")
                print(f"  Vote counts: {dict(vote_counts)}")
        else:
            # No smoothing, use raw prediction
            action_idx = raw_action_idx

        # DEBUG: Print prediction details
        if config.DEBUG_MODE:
            probs = self.model.predict_probs(state_tensor)
            print(f"\n[DEBUG] AI Prediction #{self.total_predictions + 1}")
            print(f"  State vector (first 5): {state_vector[:5]}")
            print(f"  Pacman pos: {game_state.get('pacman_pos')}, dir: {game_state.get('pacman_direction')}")
            print(f"  Model probabilities:")
            for i, prob in enumerate(probs):
                action_str = config.IDX_TO_ACTION[i]
                marker = " <-- SELECTED" if i == action_idx else ""
                print(f"    {action_str:8s}: {prob:.4f}{marker}")

        # Track statistics (provided)
        self.total_predictions += 1
        action_name = config.IDX_TO_ACTION[action_idx]
        self.action_counts[action_name] += 1

        if return_probs:
            probs = self.model.predict_probs(state_tensor)
            self.confidence_scores.append(probs.max().item())
            return action_idx, probs
        else:
            return action_idx

    def get_action_name(self, game_state: dict) -> str:
        """
        Get action name instead of index (for easier integration)

        Args:
            game_state: Dictionary with current game state

        Returns:
            action_name: Action name ("UP", "DOWN", "LEFT", "RIGHT", "NONE")
        """
        action_idx = self.get_action(game_state)
        return config.IDX_TO_ACTION[action_idx]

    def print_statistics(self):
        """Print statistics about AI predictions"""
        print("\n" + "=" * 70)
        print("AI PLAYER STATISTICS")
        print("=" * 70)
        print(f"Total predictions: {self.total_predictions:,}")
        print("\nAction Distribution:")

        for action, count in self.action_counts.items():
            if self.total_predictions > 0:
                percentage = (count / self.total_predictions) * 100
                bar = "█" * int(percentage / 2)
                print(f"  {action:8s}: {count:6d} ({percentage:5.2f}%) {bar}")

        if len(self.confidence_scores) > 0:
            avg_confidence = np.mean(self.confidence_scores)
            print(f"\nAverage confidence: {avg_confidence:.4f}")

        print("=" * 70)

    def reset_statistics(self):
        """Reset statistics counters"""
        self.action_counts = {action: 0 for action in config.ACTION_TO_IDX.keys()}
        self.total_predictions = 0
        self.confidence_scores = []
        self.prediction_buffer = []  # Clear prediction buffer


# ============================================================================
# GAME INTEGRATION (For students to connect with Pac-Man game)
# ============================================================================

class GameIntegration:
    """
    Helper class to integrate AI player with the Pac-Man game

    Students will use this to replace keyboard input with model predictions
    """

    def __init__(self):
        self.ai_player = AIPlayer()
        self.game_state_cache = None
        self.show_predictions = config.SHOW_PREDICTIONS

    def update_game_state(self, game_state: dict):
        """
        Update cached game state

        Args:
            game_state: Current game state from Pac-Man
        """
        self.game_state_cache = game_state

    def get_ai_action(self) -> str:
        """
        Get AI action for current game state

        Returns:
            action_name: Action to take ("UP", "DOWN", "LEFT", "RIGHT", "NONE")
        """
        if self.game_state_cache is None:
            return "NONE"

        # Get action with probabilities
        action_idx, probs = self.ai_player.get_action(
            self.game_state_cache,
            return_probs=True
        )
        action_name = config.IDX_TO_ACTION[action_idx]

        # Print predictions if enabled
        if self.show_predictions and self.ai_player.total_predictions % 60 == 0:
            self._print_prediction(action_name, probs)

        return action_name

    def _print_prediction(self, action: str, probs: torch.Tensor):
        """Print current prediction"""
        print(f"\n[AI] Predicted action: {action}")
        print("  Probabilities:")
        for i, prob in enumerate(probs):
            action_name = config.IDX_TO_ACTION[i]
            bar = "█" * int(prob * 20)
            print(f"    {action_name:8s}: {prob:.3f} {bar}")


# ============================================================================
# TESTING / DEMO
# ============================================================================

def test_ai_player():
    """Test AI player with dummy game states"""
    print("\n" + "=" * 70)
    print("TESTING AI PLAYER")
    print("=" * 70)

    # Check if model exists
    if not config.MODEL_SAVE_PATH.exists():
        print(f"\nERROR: No trained model found at {config.MODEL_SAVE_PATH}")
        print("Please train a model first using train.py")
        return

    # Create AI player
    ai_player = AIPlayer()

    # Create dummy game states and test
    print("\nTesting with dummy game states...\n")

    test_states = [
        {
            "pacman_pos": (14, 23),
            "pacman_direction": 0,
            "ghosts": [
                {"pos": (11, 14), "state": "SCATTER"},
                {"pos": (13, 14), "state": "CHASE"},
                {"pos": (15, 14), "state": "CHASE"},
                {"pos": (13, 11), "state": "FRIGHTENED"}
            ],
            "nearest_seed": (14, 22)
        },
        {
            "pacman_pos": (10, 10),
            "pacman_direction": 2,
            "ghosts": [
                {"pos": (11, 10), "state": "CHASE"},
                {"pos": (12, 10), "state": "CHASE"},
                {"pos": (15, 14), "state": "SCATTER"},
                {"pos": (13, 11), "state": "SCATTER"}
            ],
            "nearest_seed": (9, 10)
        }
    ]

    for i, game_state in enumerate(test_states):
        print(f"Test case {i+1}:")
        print(f"  Pacman at {game_state['pacman_pos']}")

        action_idx, probs = ai_player.get_action(game_state, return_probs=True)
        action_name = config.IDX_TO_ACTION[action_idx]

        print(f"  Predicted action: {action_name}")
        print(f"  Confidence: {probs.max().item():.4f}")
        print()

    # Print statistics
    ai_player.print_statistics()

    print("\n✓ AI Player test complete!")
    print("\nNext steps:")
    print("  1. Integrate with the Pac-Man game (modify game files)")
    print("  2. Run the game with AI control enabled")
    print("  3. Compare AI performance with your own gameplay")


def main():
    """Main function for auto-play mode"""
    print("\n" + "=" * 70)
    print("PAC-MAN BEHAVIOR CLONING - AUTO-PLAY MODE")
    print("=" * 70)

    # Check if model exists
    if not config.MODEL_SAVE_PATH.exists():
        print(f"\n❌ ERROR: No trained model found!")
        print(f"Expected path: {config.MODEL_SAVE_PATH}")
        print("\nPlease train a model first:")
        print("  python train.py")
        return

    # Run test
    test_ai_player()

    print("\n" + "=" * 70)
    print("INTEGRATION INSTRUCTIONS")
    print("=" * 70)
    print("""
To enable AI auto-play in the Pac-Man game:

1. Import the GameIntegration class in your game code:
   from auto_play import GameIntegration

2. Create an instance:
   ai_integration = GameIntegration()

3. In your game loop:
   - Extract current game state
   - Update: ai_integration.update_game_state(game_state)
   - Get action: action = ai_integration.get_ai_action()
   - Apply action to Pac-Man

4. Run the game and watch your model play!
    """)
    print("=" * 70)


if __name__ == "__main__":
    main()
