"""
Configuration file for Pac-Man Behavior Cloning Assignment

This file contains all hyperparameters and settings for the assignment.
Students can experiment with these values to improve model performance.
"""

import torch
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR = Path(__file__).parent
TRAJECTORY_DIR = BASE_DIR / "trajectories"
MODEL_DIR = BASE_DIR / "models"

# Ensure directories exist
TRAJECTORY_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# ============================================================================
# GAME SETTINGS
# ============================================================================
GAME_FPS = 60  # Pac-Man runs at 60 FPS
RECORD_EVERY_N_FRAMES = 1  # Record every frame (students can change to sample less frequently)
INFERENCE_EVERY_N_FRAMES = 1  # AI predicts action every N frames (1 = 60 predictions/sec, 2 = 30/sec, etc.)

# ============================================================================
# STATE REPRESENTATION
# ============================================================================
STATE_DIM = 70  # Total number of features (comprehensive feature set, action history removed)

# Feature breakdown (70 total):
# BASIC FEATURES:
# - Pacman position (x, y): 2
# - Pacman direction (one-hot): 4
# - 4 Ghost positions (x, y): 8
# - 4 Ghost states (one-hot: chase/scatter/frightened): 12
# - Nearest seed direction (dx, dy): 2
#
# ESSENTIAL FEATURES (1-4):
# - Ghost distances (4 ghosts): 4
# - Wall adjacency (4 directions): 4
# - Distance to nearest energizer: 1
#
# HIGH VALUE FEATURES (5-8):
# - Lives remaining: 1
# - Frightened ghost count: 1
# - Ghost relative directions (4 ghosts × 4 dirs): 16
# - Valid movement directions: 4
#
# ADVANCED FEATURES (9-11):
# - Local seed density: 1
# - Predicted ghost positions (4 ghosts × 2): 8
# - Trap detection (danger score, escape routes): 2
# NOTE: Action history (3) removed to prevent feedback loop during inference
#
# Total: 2+4+8+12+2 + 4+4+1 + 1+1+16+4 + 1+8+2 = 70

# Grid dimensions for normalization
GRID_WIDTH = 28   # 28 cells wide
GRID_HEIGHT = 31  # 31 cells tall

# ============================================================================
# ACTION SPACE
# ============================================================================
NUM_ACTIONS = 5

# Action mapping
ACTION_TO_IDX = {
    "UP": 0,
    "DOWN": 1,
    "LEFT": 2,
    "RIGHT": 3,
    "NONE": 4  # No action / continue current direction
}

IDX_TO_ACTION = {v: k for k, v in ACTION_TO_IDX.items()}

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================
# Students can experiment with these
HIDDEN_DIMS = [128, 64]  # Hidden layer sizes
DROPOUT_RATE = 0.2
ACTIVATION = "relu"  # Options: "relu", "tanh", "leaky_relu"

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================
# Students can tune these for better performance
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 1000
WEIGHT_DECAY = 1e-5  # L2 regularization

# Data split
TRAIN_SPLIT = 0.8  # 80% train, 20% validation

# Early stopping
PATIENCE = 10  # Stop if no improvement for N epochs
MIN_DELTA = 0.001  # Minimum improvement to count as progress

# ============================================================================
# DEVICE SETTINGS
# ============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# LOGGING
# ============================================================================
LOG_INTERVAL = 10  # Print training stats every N batches
SAVE_BEST_MODEL = True
MODEL_SAVE_PATH = MODEL_DIR / "best_pacman_model.pth"

# ============================================================================
# EVALUATION
# ============================================================================
# Number of episodes to run for evaluation
EVAL_EPISODES = 5

# Visualization
SHOW_PREDICTIONS = True  # Display model predictions during auto-play
DEBUG_MODE = False  # Print detailed debugging info during AI play

# Prediction smoothing
USE_SLIDING_WINDOW = True  # Use majority voting over recent predictions
SLIDING_WINDOW_SIZE = 5  # Number of recent predictions to consider (odd number recommended)

# ============================================================================
# ADVANCED (Optional for students to explore)
# ============================================================================
# Data augmentation
USE_DATA_AUGMENTATION = False  # Flip left/right actions
AUGMENTATION_PROB = 0.5

# Class balancing
USE_CLASS_WEIGHTS = True  # Weight loss by inverse class frequency
BALANCE_DATASET = False  # Oversample minority actions (use ONE method at a time)

# Curriculum learning
USE_CURRICULUM = False  # Start with shorter trajectories

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def print_config():
    """Print current configuration settings"""
    print("=" * 70)
    print("PAC-MAN BEHAVIOR CLONING - CONFIGURATION")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"State dimension: {STATE_DIM}")
    print(f"Number of actions: {NUM_ACTIONS}")
    print(f"\nModel Architecture:")
    print(f"  Hidden layers: {HIDDEN_DIMS}")
    print(f"  Dropout: {DROPOUT_RATE}")
    print(f"  Activation: {ACTIVATION}")
    print(f"\nTraining Settings:")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Train/Val split: {TRAIN_SPLIT:.0%}/{1-TRAIN_SPLIT:.0%}")
    print(f"\nPaths:")
    print(f"  Trajectories: {TRAJECTORY_DIR}")
    print(f"  Models: {MODEL_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    print_config()
