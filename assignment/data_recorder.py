"""
Data Recorder for Pac-Man Behavior Cloning

This module records gameplay trajectories (state-action pairs) while you play Pac-Man.
Students will implement key functions to extract game state features and record actions.

Usage:
    recorder = GameplayRecorder()

    # In game loop:
    state = extract_state_features(game_objects)
    action = map_keyboard_to_action(key_event)
    recorder.record_step(state, action)

    # After game ends:
    recorder.save_trajectory()
"""

import numpy as np
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import config


class GameplayRecorder:
    """Records gameplay trajectories for behavior cloning"""

    def __init__(self):
        """Initialize the recorder"""
        self.states = []
        self.actions = []
        self.metadata = {
            "start_time": datetime.now().isoformat(),
            "frame_count": 0,
            "game_score": 0,
            "lives_remaining": 3,
            "level": 1,
            "difficulty": "NORMAL"
        }
        self.recording = False

    def start_recording(self, level: int = 1, difficulty: str = "NORMAL"):
        """Start a new recording session"""
        self.states = []
        self.actions = []
        self.metadata = {
            "start_time": datetime.now().isoformat(),
            "frame_count": 0,
            "game_score": 0,
            "lives_remaining": 3,
            "level": level,
            "difficulty": difficulty
        }
        self.recording = True
        print(f"[Recorder] Started recording - Level {level}, Difficulty {difficulty}")

    def record_step(self, state: np.ndarray, action: int):
        """
        Record a single (state, action) pair

        Args:
            state: State feature vector (shape: [STATE_DIM])
            action: Action index (0-4)
        """
        if not self.recording:
            return

        self.states.append(state)
        self.actions.append(action)
        self.metadata["frame_count"] += 1

    def stop_recording(self, final_score: int, lives_remaining: int):
        """Stop recording and update metadata"""
        if not self.recording:
            return

        self.metadata["end_time"] = datetime.now().isoformat()
        self.metadata["game_score"] = final_score
        self.metadata["lives_remaining"] = lives_remaining
        self.recording = False

        print(f"[Recorder] Stopped recording - {len(self.states)} frames, Score: {final_score}")

    def save_trajectory(self, filename: Optional[str] = None) -> Path:
        """
        Save recorded trajectory to disk

        Args:
            filename: Optional custom filename. If None, uses timestamp

        Returns:
            Path to saved file
        """
        if len(self.states) == 0:
            print("[Recorder] Warning: No data to save!")
            return None

        # Convert lists to numpy arrays
        states_array = np.array(self.states, dtype=np.float32)
        actions_array = np.array(self.actions, dtype=np.int64)

        # Create filename
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trajectory_{timestamp}.pkl"

        filepath = config.TRAJECTORY_DIR / filename

        # Save data
        data = {
            "states": states_array,
            "actions": actions_array,
            "metadata": self.metadata
        }

        with open(filepath, "wb") as f:
            pickle.dump(data, f)

        print(f"[Recorder] Saved trajectory to {filepath}")
        print(f"  - Frames: {len(states_array)}")
        print(f"  - State shape: {states_array.shape}")
        print(f"  - Actions shape: {actions_array.shape}")

        return filepath

    def get_statistics(self) -> Dict:
        """Get statistics about recorded data"""
        if len(self.actions) == 0:
            return {}

        actions_array = np.array(self.actions)
        unique, counts = np.unique(actions_array, return_counts=True)

        stats = {
            "total_frames": len(self.actions),
            "action_distribution": {
                config.IDX_TO_ACTION[int(idx)]: int(count)
                for idx, count in zip(unique, counts)
            },
            "duration_seconds": self.metadata["frame_count"] / config.GAME_FPS,
        }

        return stats


# ============================================================================
# STATE EXTRACTION FUNCTIONS (Students implement these)
# ============================================================================

def extract_state_features(game_state: Dict) -> np.ndarray:
    """
    Extract comprehensive state features from the game for the neural network

    Complete implementation with all 12 feature categories for optimal learning.

    Args:
        game_state: Dictionary containing all game state information

    Returns:
        state_vector: np.ndarray of shape [STATE_DIM]

    Feature breakdown (73 features total):
        ESSENTIAL (1-4):
        - Pacman position (2)
        - Pacman direction one-hot (4)
        - Ghost positions (8)
        - Ghost states (12)
        - Nearest seed direction (2)
        - Ghost distances (4)
        - Wall adjacency (4)
        - Distance to nearest energizer (1)

        HIGH VALUE (5-8):
        - Lives remaining (1)
        - Frightened ghost count (1)
        - Ghost relative directions (16)
        - Valid movement directions (4)

        ADVANCED (9-12):
        - Local seed density (1)
        - Action history (3)
        - Predicted ghost positions (8)
        - Trap detection (2)
    """
    from feature_utils import (
        euclidean_distance, manhattan_distance, get_wall_adjacency,
        get_valid_moves, calculate_seed_density, predict_ghost_position,
        detect_trap_situation, get_ghost_relative_direction
    )

    pacman_pos = game_state["pacman_pos"]
    pacman_x, pacman_y = pacman_pos
    pacman_dir = game_state["pacman_direction"]

    # ========================================================================
    # BASIC FEATURES (same as before)
    # ========================================================================

    # 1. Pacman position (2 features)
    pacman_x_norm = pacman_x / config.GRID_WIDTH
    pacman_y_norm = pacman_y / config.GRID_HEIGHT

    # 2. Pacman direction one-hot (4 features)
    direction_onehot = np.zeros(4)
    if 0 <= pacman_dir < 4:
        direction_onehot[pacman_dir] = 1

    # 3. Ghost positions (8 features)
    ghost_positions = []
    for ghost in game_state["ghosts"]:
        gx, gy = ghost["pos"]
        gx_norm = gx / config.GRID_WIDTH
        gy_norm = gy / config.GRID_HEIGHT
        ghost_positions.extend([gx_norm, gy_norm])
    ghost_positions = np.array(ghost_positions)

    # 4. Ghost states (12 features)
    ghost_state_mapping = {"CHASE": 0, "SCATTER": 1, "FRIGHTENED": 2}
    ghost_states = []
    for ghost in game_state["ghosts"]:
        state_str = ghost["state"]
        state_onehot = np.zeros(3)
        if state_str in ghost_state_mapping:
            state_idx = ghost_state_mapping[state_str]
            state_onehot[state_idx] = 1
        ghost_states.extend(state_onehot)
    ghost_states = np.array(ghost_states)

    # 5. Nearest seed direction (2 features)
    if game_state.get("nearest_seed") is not None:
        seed_x, seed_y = game_state["nearest_seed"]
        dx = (seed_x - pacman_x) / config.GRID_WIDTH
        dy = (seed_y - pacman_y) / config.GRID_HEIGHT
    else:
        dx, dy = 0.0, 0.0
    seed_direction = np.array([dx, dy])

    # ========================================================================
    # ESSENTIAL FEATURES (1-4)
    # ========================================================================

    # 6. Ghost distances (4 features)
    ghost_distances = []
    max_dist = (config.GRID_WIDTH**2 + config.GRID_HEIGHT**2) ** 0.5
    for ghost in game_state["ghosts"]:
        dist = euclidean_distance(pacman_pos, ghost["pos"])
        ghost_distances.append(dist / max_dist)  # Normalize
    ghost_distances = np.array(ghost_distances)

    # 7. Wall adjacency (4 features)
    collision_map = game_state.get("collision_map")
    if collision_map is not None:
        walls = get_wall_adjacency(pacman_pos, collision_map)
        wall_features = np.array(walls, dtype=float)
    else:
        wall_features = np.zeros(4)

    # 8. Distance to nearest energizer (1 feature)
    if game_state.get("nearest_energizer") is not None:
        energizer_dist = euclidean_distance(pacman_pos, game_state["nearest_energizer"])
        energizer_dist_norm = energizer_dist / max_dist
    else:
        energizer_dist_norm = 1.0  # No energizer available
    energizer_feature = np.array([energizer_dist_norm])

    # ========================================================================
    # HIGH VALUE FEATURES (5-8)
    # ========================================================================

    # 9. Lives remaining (1 feature)
    lives = game_state.get("lives", 3)
    lives_norm = lives / 3.0  # Normalize by typical max lives
    lives_feature = np.array([lives_norm])

    # 10. Frightened ghost count (1 feature)
    frightened_count = sum(1 for ghost in game_state["ghosts"] if ghost["state"] == "FRIGHTENED")
    frightened_norm = frightened_count / 4.0  # Normalize by total ghosts
    frightened_feature = np.array([frightened_norm])

    # 11. Ghost relative directions (16 features = 4 ghosts × 4 directions)
    ghost_relative_dirs = []
    for ghost in game_state["ghosts"]:
        rel_dir = get_ghost_relative_direction(pacman_pos, ghost["pos"])
        ghost_relative_dirs.extend(rel_dir)
    ghost_relative_dirs = np.array(ghost_relative_dirs)

    # 12. Valid movement directions (4 features)
    if collision_map is not None:
        valid_moves = get_valid_moves(pacman_pos, collision_map)
        valid_move_features = np.array(valid_moves, dtype=float)
    else:
        valid_move_features = np.ones(4)  # Assume all valid if no map

    # ========================================================================
    # ADVANCED FEATURES (9-12)
    # ========================================================================

    # 13. Local seed density (1 feature)
    seeds_map = game_state.get("seeds_map")
    if seeds_map is not None:
        seed_density = calculate_seed_density(pacman_pos, seeds_map, radius=5)
    else:
        seed_density = 0.5  # Default
    seed_density_feature = np.array([seed_density])

    # 14. Action history (3 features) - DISABLED to prevent feedback loop during inference
    # The model would see "last actions were RIGHT" and predict RIGHT again, creating infinite loop
    # action_history = game_state.get("action_history", [])
    # history_length = 3
    # recent_actions = action_history[-history_length:] if action_history else []
    # while len(recent_actions) < history_length:
    #     recent_actions.insert(0, 4)  # NONE = 4
    # action_history_features = np.array(recent_actions) / config.NUM_ACTIONS

    # 15. Predicted ghost positions (8 features = 4 ghosts × 2 coords)
    predicted_ghost_positions = []
    for ghost in game_state["ghosts"]:
        ghost_dir = ghost.get("direction")
        predicted_pos = predict_ghost_position(ghost["pos"], ghost_dir, steps=2)
        pred_x_norm = predicted_pos[0] / config.GRID_WIDTH
        pred_y_norm = predicted_pos[1] / config.GRID_HEIGHT
        predicted_ghost_positions.extend([pred_x_norm, pred_y_norm])
    predicted_ghost_positions = np.array(predicted_ghost_positions)

    # 16. Trap detection (2 features: danger score, escape routes)
    ghost_positions_list = [ghost["pos"] for ghost in game_state["ghosts"]]
    if collision_map is not None:
        danger_score, escape_routes = detect_trap_situation(
            pacman_pos, ghost_positions_list, collision_map, danger_radius=5
        )
        escape_routes_norm = escape_routes / 4.0  # Normalize by max possible
    else:
        danger_score, escape_routes_norm = 0.0, 1.0
    trap_features = np.array([danger_score, escape_routes_norm])

    # ========================================================================
    # CONCATENATE ALL FEATURES
    # ========================================================================

    state_vector = np.concatenate([
        [pacman_x_norm, pacman_y_norm],      # 2
        direction_onehot,                     # 4
        ghost_positions,                      # 8
        ghost_states,                         # 12
        seed_direction,                       # 2
        ghost_distances,                      # 4
        wall_features,                        # 4
        energizer_feature,                    # 1
        lives_feature,                        # 1
        frightened_feature,                   # 1
        ghost_relative_dirs,                  # 16
        valid_move_features,                  # 4
        seed_density_feature,                 # 1
        # action_history_features,            # 3 - REMOVED (feedback loop)
        predicted_ghost_positions,            # 8
        trap_features                         # 2
    ])

    # Total: 2+4+8+12+2+4+4+1+1+1+16+4+1+8+2 = 70 features (was 73, removed 3 action history)

    assert state_vector.shape == (config.STATE_DIM,), \
        f"State vector has wrong shape: {state_vector.shape}, expected ({config.STATE_DIM},)"

    return state_vector.astype(np.float32)


def map_keyboard_to_action(key_event: str) -> int:
    """
    Map keyboard input to action index

    Args:
        key_event: String representing key pressed ("UP", "DOWN", "LEFT", "RIGHT", or "NONE")

    Returns:
        action_idx: Integer 0-4
    """
    return config.ACTION_TO_IDX.get(key_event, config.ACTION_TO_IDX["NONE"])


# ============================================================================
# UTILITY FUNCTIONS (Provided - students don't need to modify)
# ============================================================================

def load_trajectory(filepath: Path) -> Dict:
    """Load a saved trajectory from disk"""
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data


def load_all_trajectories() -> List[Dict]:
    """Load all trajectories from the trajectories directory"""
    trajectories = []

    for filepath in config.TRAJECTORY_DIR.glob("*.pkl"):
        try:
            data = load_trajectory(filepath)

            # Skip initial NONE actions (waiting at game start)
            # This prevents the AI from getting stuck at the starting position
            actions = data['actions']
            skip_count = 0
            for action in actions:
                if action == 4:  # NONE action
                    skip_count += 1
                else:
                    break

            if skip_count > 0:
                # Remove initial NONE frames
                data['states'] = data['states'][skip_count:]
                data['actions'] = data['actions'][skip_count:]
                print(f"Loaded {filepath.name}: {len(data['states'])} frames (skipped {skip_count} initial NONE)")
            else:
                print(f"Loaded {filepath.name}: {len(data['states'])} frames")

            # MIGRATION: Remove action history features (indices 60-62) if old 73-feature trajectories
            if data['states'].shape[1] == 73:
                # Remove features 60, 61, 62 (action history)
                # Keep [0:60] and [63:73]
                states_part1 = data['states'][:, :60]  # Features 0-59
                states_part2 = data['states'][:, 63:]  # Features 63-72
                data['states'] = np.concatenate([states_part1, states_part2], axis=1)
                print(f"  -> Migrated from 73 to 70 features (removed action history)")

            # Only add if there's still data left
            if len(data['states']) > 0:
                trajectories.append(data)
        except Exception as e:
            print(f"Error loading {filepath.name}: {e}")

    print(f"\nTotal trajectories loaded: {len(trajectories)}")
    return trajectories


def print_trajectory_stats(trajectories: List[Dict]):
    """Print statistics about loaded trajectories"""
    if len(trajectories) == 0:
        print("No trajectories to analyze!")
        return

    total_frames = sum(len(t["states"]) for t in trajectories)
    total_score = sum(t["metadata"]["game_score"] for t in trajectories)

    # Aggregate action distribution
    all_actions = np.concatenate([t["actions"] for t in trajectories])
    unique, counts = np.unique(all_actions, return_counts=True)

    print("\n" + "=" * 70)
    print("TRAJECTORY STATISTICS")
    print("=" * 70)
    print(f"Number of trajectories: {len(trajectories)}")
    print(f"Total frames: {total_frames:,}")
    print(f"Average frames per trajectory: {total_frames / len(trajectories):.1f}")
    print(f"Total score: {total_score:,}")
    print(f"Average score: {total_score / len(trajectories):.1f}")
    print(f"\nAction Distribution:")
    for idx, count in zip(unique, counts):
        action_name = config.IDX_TO_ACTION[int(idx)]
        percentage = (count / len(all_actions)) * 100
        print(f"  {action_name:8s}: {count:6d} ({percentage:5.2f}%)")
    print("=" * 70)


# ============================================================================
# TESTING / DEMO
# ============================================================================

if __name__ == "__main__":
    # Demo: Test state extraction with dummy data
    print("Testing comprehensive state extraction...\n")

    # Create dummy collision map (all passable except edges)
    dummy_collision_map = [[True] * config.GRID_WIDTH for _ in range(config.GRID_HEIGHT)]
    for i in range(config.GRID_WIDTH):
        dummy_collision_map[0][i] = False  # Top wall
        dummy_collision_map[config.GRID_HEIGHT - 1][i] = False  # Bottom wall
    for i in range(config.GRID_HEIGHT):
        dummy_collision_map[i][0] = False  # Left wall
        dummy_collision_map[i][config.GRID_WIDTH - 1] = False  # Right wall

    # Create dummy seeds map
    dummy_seeds_map = [[True] * config.GRID_WIDTH for _ in range(config.GRID_HEIGHT)]
    # Clear some areas
    for i in range(10, 18):
        for j in range(10, 18):
            dummy_seeds_map[j][i] = False

    dummy_game_state = {
        "pacman_pos": (14, 23),
        "pacman_direction": 0,  # UP
        "ghosts": [
            {"pos": (11, 14), "state": "SCATTER", "direction": 0},
            {"pos": (13, 14), "state": "CHASE", "direction": 1},
            {"pos": (15, 14), "state": "CHASE", "direction": 2},
            {"pos": (13, 11), "state": "FRIGHTENED", "direction": None}
        ],
        "collision_map": dummy_collision_map,
        "seeds_map": dummy_seeds_map,
        "seed_positions": [(14, 22), (14, 21), (14, 20)],
        "energizer_positions": [(1, 1), (26, 1)],
        "nearest_seed": (14, 22),
        "nearest_energizer": (1, 1),
        "lives": 3,
        "score": 1000,
        "action_history": [0, 2, 3]  # UP, LEFT, RIGHT
    }

    try:
        state = extract_state_features(dummy_game_state)
        print(f"✓ State vector shape: {state.shape}")
        print(f"✓ Expected shape: ({config.STATE_DIM},)")
        print(f"✓ Feature extraction successful!")
        print(f"\nFirst 10 features: {state[:10]}")
        print(f"Last 10 features: {state[-10:]}")

        # Test action mapping
        print("\n" + "=" * 70)
        print("Testing action mapping...")
        for key in ["UP", "DOWN", "LEFT", "RIGHT", "NONE"]:
            action_idx = map_keyboard_to_action(key)
            print(f"  {key:8s} -> {action_idx}")

        print("\n" + "=" * 70)
        print("✓ ALL TESTS PASSED!")
        print("=" * 70)
        print("\nComprehensive feature extraction with 73 features:")
        print("  - Basic features: 28")
        print("  - Essential features: 9")
        print("  - High value features: 22")
        print("  - Advanced features: 14")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\n" + "=" * 70)
        print("Fix the errors above and try again!")
        print("=" * 70)
