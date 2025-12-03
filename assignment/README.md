# Pac-Man Behavior Cloning: Hands-On Learning

**Goal:** Experience a complete machine learning pipeline by training an AI to play Pac-Man using behavior cloning.

**Format:** Hands-on session where you record gameplay, train a neural network, and watch it play.

---

## Table of Contents

1. [Overview](#overview)
2. [Learning Objectives](#learning-objectives)
3. [Quick Start](#quick-start)
4. [Hands-On Workflow](#hands-on-workflow)
5. [Understanding the Code](#understanding-the-code)
6. [Expected Results](#expected-results)
7. [Tips & Tricks](#tips--tricks)
8. [Troubleshooting](#troubleshooting)
9. [Going Further](#going-further)

---

## Overview

This is a **complete, working implementation** of behavior cloning for Pac-Man. You will:

1. **Play Pac-Man** and record your gameplay as (state, action) pairs
2. **Train a neural network** to imitate your playing style
3. **Watch the AI play** and analyze its behavior
4. **Learn** about the ML pipeline, distribution shift, and model deployment

This is a practical introduction to **behavior cloning** (imitation learning), where an AI learns to mimic expert demonstrations. All code is fully implemented - your focus is on **using** the pipeline and **understanding** how it works.

---

## Quick Start

### Prerequisites

- Python 3.8+
- Git

### Installation

```bash
# Clone repository (if not already done)
git clone git@github.com:KwonVitalLab/pacman-bmi534-imitation-learning.git pacman-bmi534-imitation-learning
cd pacman-bmi534-imitation-learning
git submodule update --init --recursive

# Navigate to assignment directory
cd assignment

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
python run_game.py --mode human
```

You should see the Pac-Man game window. Play a quick game to verify everything works!

**Controls:**
- Arrow keys or WASD to move
- ESC to pause
- Close window to quit

---

## Hands-On Workflow

### Step 1: Record Your Gameplay (10 minutes)

Play Pac-Man while the system records your (state, action) pairs:

```bash
python run_game.py --mode record
```

**Guidelines:**
- Play **5-10 complete games** (each ~2 minutes)
- Try to **survive as long as possible** (more data!)
- Use **different strategies** (aggressive, defensive, etc.)
- Each game generates ~1000-2000 frames

**What's happening behind the scenes:**
- Every frame, the system extracts a 70-dimensional state vector
- Your keyboard input is recorded as an action (0-4)
- Data is saved to `trajectories/trajectory_YYYYMMDD_HHMMSS.pkl`

### Step 2: Inspect Your Data (2 minutes)

Check what you recorded:

```bash
python dataset.py
```

You'll see:
- Total number of frames collected
- Action distribution (how often you pressed each direction)
- Dataset statistics

**Think about:** Are some actions rare? This creates "class imbalance" - the model won't learn rare actions well.

### Step 3: Train the Model (10-15 minutes)

Train a neural network to imitate your gameplay:

```bash
python train.py
```

**What's happening:**
- Dataset is split 80% train / 20% validation
- 3-layer MLP trains to predict actions from states
- Training runs for up to 1000 epochs (with early stopping)
- Best model is saved to `models/best_pacman_model.pth`

**What to expect:**
- Training takes 5-15 minutes on CPU
- Validation accuracy typically reaches 60-85%
- Higher accuracy = better imitation (but not guaranteed better gameplay!)

**Monitor:**
- Loss should decrease over time
- Validation accuracy should plateau
- Per-class accuracy shows which actions are hard to learn

### Step 4: Watch AI Play (5+ minutes)

See your trained model play Pac-Man:

```bash
python run_game.py --mode ai
```

**Observe:**
- Does it play like you?
- Where does it succeed? Where does it fail?
- Does it get stuck in corners or loops?
- How does it handle ghosts?

**Key questions to discuss:**
- Why does the AI fail in certain situations?
- What states did it encounter that weren't in training data?
- How would you improve performance?

---

## Understanding the Code

### The Pipeline

```
┌─────────────────────────────────────────────────────┐
│  RECORD PHASE                                       │
├─────────────────────────────────────────────────────┤
│  run_game.py --mode record                          │
│         ↓                                           │
│  data_recorder.py (extracts 70 state features)      │
│         ↓                                           │
│  trajectories/trajectory_*.pkl                      │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  TRAINING PHASE                                     │
├─────────────────────────────────────────────────────┤
│  train.py                                           │
│         ↓                                           │
│  dataset.py (loads data, creates DataLoader)        │
│         ↓                                           │
│  model.py (3-layer MLP: 70 → 128 → 64 → 5)         │
│         ↓                                           │
│  Cross-entropy loss + Adam optimizer                │
│         ↓                                           │
│  models/best_pacman_model.pth                       │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  INFERENCE PHASE                                    │
├─────────────────────────────────────────────────────┤
│  run_game.py --mode ai                              │
│         ↓                                           │
│  auto_play.py (loads model, predicts actions)       │
│         ↓                                           │
│  Game receives AI's predicted actions               │
│         ↓                                           │
│  🎮 AI plays Pac-Man!                               │
└─────────────────────────────────────────────────────┘
```

### Key Files

| File | Purpose | Key Concepts |
|------|---------|--------------|
| **config.py** | Hyperparameters | Learning rate, batch size, network architecture |
| **data_recorder.py** | Feature extraction | State representation (70 features), normalization |
| **dataset.py** | Data loading | PyTorch Dataset, normalization statistics, class balancing |
| **model.py** | Neural network | MLP architecture, forward pass, dropout |
| **train.py** | Training loop | Loss calculation, backpropagation, early stopping |
| **auto_play.py** | AI inference | Model loading, prediction, sliding window voting |
| **run_game.py** | Game integration | Recording mode, AI mode, human mode |
| **feature_utils.py** | Advanced features | BFS pathfinding, trap detection, ghost prediction |

### The State Representation (70 features)

The AI doesn't see raw pixels - it sees a carefully engineered feature vector:

**Basic Features (28):**
- Pac-Man position (x, y) - normalized
- Pac-Man direction - one-hot encoded (4 dims)
- 4 Ghost positions (x, y each) - normalized
- 4 Ghost states (normal/frightened) - one-hot encoded
- Nearest seed position

**Distance Features (9):**
- Distance to each of 4 ghosts
- Distance to nearest energizer
- Wall adjacency in 4 directions

**Strategic Features (22):**
- Lives remaining
- Number of frightened ghosts
- Ghost relative directions - one-hot encoded
- Valid moves in 4 directions

**Advanced Features (11):**
- Seed density in local area
- Predicted ghost positions (1 step ahead)
- Trap detection (dead end identification)

**Why features instead of pixels?**
- Much smaller input space (70 vs 100,000+ pixels)
- Faster training
- Easier to interpret
- Better for behavior cloning (limited data)

### The Neural Network

```
Input: 70 features
   ↓
Linear Layer: 70 → 128 neurons
   ↓
ReLU Activation
   ↓
Dropout: 20% (prevents overfitting)
   ↓
Linear Layer: 128 → 64 neurons
   ↓
ReLU Activation
   ↓
Dropout: 20%
   ↓
Linear Layer: 64 → 5 neurons
   ↓
Output: 5 action probabilities (UP, DOWN, LEFT, RIGHT, NONE)
```

**Total parameters:** ~11,269
**Training:** Supervised learning with cross-entropy loss
**Inference:** Argmax over output probabilities (with sliding window smoothing)

---

## Troubleshooting

### Installation Issues

**Problem:** Pygame won't install

```bash
# On Mac with M1/M2 chip:
pip install pygame --pre

# On Linux:
sudo apt-get install python3-pygame
```

**Problem:** PyTorch is too large

```bash
# Install CPU-only version:
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Runtime Issues

**Problem:** "No module named 'pacman'"

Make sure you're in the `assignment/` directory and initialized submodules:
```bash
cd assignment
git submodule update --init --recursive
```

**Problem:** Game window doesn't appear

Check Pygame installation:
```bash
python -c "import pygame; print(pygame.version.ver)"
```

**Problem:** "No trajectories found"

Record some games first:
```bash
python run_game.py --mode record
# Play at least one complete game
ls trajectories/  # Should show .pkl files
```

**Problem:** "No trained model found"

Train the model:
```bash
python train.py
```

### Low Performance

**Accuracy < 50%:**
1. Check data quality: `python dataset.py`
2. Record more games (10-20 total)
3. Verify action distribution isn't too imbalanced

**AI does nothing:**
- Class imbalance - model always predicts most common action
- Enable class weighting: Set `USE_CLASS_WEIGHTS = True` in `config.py`

**AI plays poorly despite high accuracy:**
- High accuracy doesn't guarantee good gameplay!
- Distribution shift: AI encounters states not in training
- Try DAgger or collect more diverse data

---

## Going Further

### Extensions & Experiments

1. **Feature Engineering:**
   - Add more state features (wall distances, predicted ghost paths)
   - Try different normalization schemes
   - Experiment with feature selection

2. **Model Architecture:**
   - Try deeper networks (more layers)
   - Try wider networks (more neurons)
   - Experiment with different activation functions
   - Add batch normalization

3. **Training Improvements:**
   - Implement data augmentation (mirror game states)
   - Try different optimizers (SGD, AdamW)
   - Experiment with learning rate schedules
   - Implement curriculum learning

4. **Advanced Techniques:**
   - **DAgger:** Iteratively collect AI's mistakes and retrain
   - **Ensemble methods:** Train multiple models and vote
   - **Hybrid BC+RL:** Fine-tune with reinforcement learning
   - **Recurrent networks:** Add LSTM to capture temporal patterns

5. **Analysis:**
   - Visualize learned features (t-SNE, PCA)
   - Analyze failure modes systematically
   - Compare different players' models
   - Create a leaderboard (longest survival time)

### Research Papers

**Foundational:**
- [A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://arxiv.org/abs/1011.0686) (DAgger)
- [End to End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316) (NVIDIA)

**Advanced:**
- [Learning from Demonstrations for Real World Reinforcement Learning](https://arxiv.org/abs/1704.03732)
- [Time-Contrastive Networks: Self-Supervised Learning from Video](https://arxiv.org/abs/1704.06888)

### Related Projects

- [Berkeley Pac-Man AI](http://ai.berkeley.edu/project_overview.html)
- [OpenAI Gym](https://gymnasium.farama.org/)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)

---

## File Structure

```
assignment/
├── README.md                  # This file
├── INSTRUCTOR_GUIDE.md        # Teaching notes
├── ASSIGNMENT_SUMMARY.md      # Technical overview
├── SOLUTIONS.md               # Implementation reference
├── requirements.txt           # Python dependencies
├── config.py                  # Hyperparameters configuration
├── data_recorder.py           # Feature extraction & recording
├── dataset.py                 # PyTorch Dataset implementation
├── model.py                   # Neural network architecture
├── train.py                   # Training loop
├── auto_play.py               # AI inference
├── run_game.py                # Game integration (3 modes)
├── feature_utils.py           # Advanced feature extraction
├── trajectories/              # Recorded gameplay (generated)
│   └── trajectory_*.pkl       # Pickle files with (state, action) pairs
└── models/                    # Trained models (generated)
    └── best_pacman_model.pth  # Best performing model checkpoint
```

---

## Discussion Questions

After completing the hands-on, consider these questions:

1. **Performance:** Why does high validation accuracy not guarantee good gameplay?

2. **Distribution Shift:** How could you measure the distribution gap between training and deployment?

3. **Data Efficiency:** How many demonstrations do you think are needed for superhuman performance?

4. **Comparison:** When would you use behavior cloning vs reinforcement learning?

5. **Real World:** What are the challenges of applying BC to autonomous driving?

6. **Ethics:** What happens if you learn from "bad" demonstrations (dangerous driving)?

---

## Acknowledgments

This project uses:
- [BaggerFast/Pacman](https://github.com/BaggerFast/Pacman) - Pygame Pac-Man implementation
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [NumPy](https://numpy.org/) - Numerical computing

---

**Enjoy exploring behavior cloning! 🎮🤖**

*Remember: The goal isn't just to get high accuracy, but to understand WHY the model succeeds and fails.*
