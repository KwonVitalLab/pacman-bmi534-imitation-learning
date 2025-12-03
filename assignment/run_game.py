"""
Game Integration Script for Pac-Man Behavior Cloning

This script wraps the Pac-Man game to add:
- Data recording mode (records your gameplay)
- AI mode (lets the trained model play)

Usage:
    python run_game.py --mode record    # Record your gameplay
    python run_game.py --mode ai        # Watch AI play
    python run_game.py --mode human     # Just play normally
"""

import sys
import os
import argparse
from pathlib import Path

# Add Pacman to path (use resolve() for absolute path - works better across Python versions)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
PACMAN_DIR = PROJECT_DIR / "Pacman"

# Debug: print paths if import fails
if not PACMAN_DIR.exists():
    print(f"Error: Pacman directory not found at {PACMAN_DIR}")
    print(f"Script dir: {SCRIPT_DIR}")
    print(f"Project dir: {PROJECT_DIR}")
    sys.exit(1)

sys.path.insert(0, str(PACMAN_DIR))

import pygame
from pacman.game import Game
from pacman.scenes.main_scene import MainScene
from pacman.scenes import SceneManager
from pacman.data_core.enums import GameStateEnum, GhostStateEnum, SoundCh
from pacman.sound import SoundController
import config
from data_recorder import GameplayRecorder, map_keyboard_to_action


class RecordingMainScene(MainScene):
    """
    Extended MainScene that records gameplay

    This class wraps the original game scene to add recording functionality
    without modifying the original Pacman code.
    """

    def __init__(self, map_color=None):
        super().__init__(map_color)
        self.recorder = GameplayRecorder()
        self.recorder.start_recording(level=1, difficulty="NORMAL")
        self.last_action = "NONE"
        self.frame_count = 0
        self.action_history = []  # Track last few actions for temporal features

        print("\n" + "=" * 70)
        print("RECORDING MODE ENABLED")
        print("=" * 70)
        print("Your gameplay will be recorded for training the neural network.")
        print("Play as best as you can!")
        print("=" * 70 + "\n")

    def get_game_state(self) -> dict:
        """
        Extract current game state for recording

        Returns:
            game_state: Dictionary with all relevant state information
        """
        # Get Pacman info
        pacman_pos = (self.pacman.rect.centerx // 8, self.pacman.rect.centery // 8)
        pacman_direction = self.pacman.rotate  # 0-3 for directions

        # Get ghost info
        ghosts = []
        for ghost in self._MainScene__ghosts:
            ghost_pos = (ghost.rect.centerx // 8, ghost.rect.centery // 8)
            ghost_direction = getattr(ghost, 'rotate', None)  # Get direction if available

            # Map ghost state to simplified state
            if ghost.state == GhostStateEnum.CHASE:
                ghost_state = "CHASE"
            elif ghost.state == GhostStateEnum.SCATTER:
                ghost_state = "SCATTER"
            elif ghost.state == GhostStateEnum.FRIGHTENED:
                ghost_state = "FRIGHTENED"
            else:
                ghost_state = "CHASE"  # Default for other states

            ghosts.append({
                "pos": ghost_pos,
                "state": ghost_state,
                "direction": ghost_direction
            })

        # Get collision map (walls) - access from loader
        collision_map = self._MainScene__loader.collision_map

        # Get all seed positions from the seed container
        seeds_map = self._MainScene__seeds._SeedContainer__seeds

        # Get remaining seed positions as list
        seed_positions = []
        for y in range(len(seeds_map)):
            for x in range(len(seeds_map[y])):
                if seeds_map[y][x]:
                    seed_positions.append((x, y))

        # Get energizer positions
        energizer_positions = [
            (energizer.x, energizer.y)
            for energizer in self._MainScene__seeds._SeedContainer__energizers
        ]

        # Find nearest seed using BFS (proper implementation)
        from feature_utils import find_nearest_item_bfs
        nearest_seed = find_nearest_item_bfs(pacman_pos, seed_positions, collision_map)

        # Find nearest energizer
        nearest_energizer = find_nearest_item_bfs(pacman_pos, energizer_positions, collision_map) if energizer_positions else None

        # Get lives and score
        lives = int(self.hp)
        score = int(self._MainScene__score)

        game_state = {
            "pacman_pos": pacman_pos,
            "pacman_direction": pacman_direction,
            "ghosts": ghosts,
            "collision_map": collision_map,
            "seeds_map": seeds_map,
            "seed_positions": seed_positions,
            "energizer_positions": energizer_positions,
            "nearest_seed": nearest_seed,
            "nearest_energizer": nearest_energizer,
            "lives": lives,
            "score": score,
            "action_history": self.action_history.copy()  # Include action history
        }

        return game_state

    def _find_nearest_seed(self, pacman_pos):
        """Find nearest seed position (simplified heuristic)"""
        # Simple heuristic: look in current direction
        px, py = pacman_pos
        direction_offsets = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # UP, DOWN, LEFT, RIGHT

        # Check in each direction for closest seed
        for dx, dy in direction_offsets:
            for dist in range(1, 10):  # Check up to 10 cells away
                check_x = px + dx * dist
                check_y = py + dy * dist

                # Bounds check
                if 0 <= check_x < config.GRID_WIDTH and 0 <= check_y < config.GRID_HEIGHT:
                    # For now, return a dummy value
                    # In a full implementation, check seeds map
                    pass

        # Return approximate nearest seed (forward direction)
        if self.pacman.rotate == 0:  # UP
            return (px, py - 1)
        elif self.pacman.rotate == 1:  # DOWN
            return (px, py + 1)
        elif self.pacman.rotate == 2:  # LEFT
            return (px - 1, py)
        else:  # RIGHT
            return (px + 1, py)

    def process_event(self, event: pygame.event.Event) -> None:
        """Override to capture keyboard input"""
        super().process_event(event)

        # Map pygame keys to action names
        if event.type == pygame.KEYDOWN:
            if event.key in [pygame.K_UP, pygame.K_w]:
                self.last_action = "UP"
                self.action_history.append(0)  # UP = 0
            elif event.key in [pygame.K_DOWN, pygame.K_s]:
                self.last_action = "DOWN"
                self.action_history.append(1)  # DOWN = 1
            elif event.key in [pygame.K_LEFT, pygame.K_a]:
                self.last_action = "LEFT"
                self.action_history.append(2)  # LEFT = 2
            elif event.key in [pygame.K_RIGHT, pygame.K_d]:
                self.last_action = "RIGHT"
                self.action_history.append(3)  # RIGHT = 3

            # Keep only last 10 actions
            if len(self.action_history) > 10:
                self.action_history = self.action_history[-10:]

    def process_logic(self) -> None:
        """Override to record state-action pairs"""
        super().process_logic()

        # Only record during active gameplay (not intro)
        if self._MainScene__state == GameStateEnum.ACTION and self.frame_count % config.RECORD_EVERY_N_FRAMES == 0:
            try:
                # Get current state
                game_state = self.get_game_state()

                # Extract state features
                from data_recorder import extract_state_features
                state_vector = extract_state_features(game_state)

                # Map action
                action_idx = map_keyboard_to_action(self.last_action)

                # Record
                self.recorder.record_step(state_vector, action_idx)

            except Exception as e:
                print(f"Warning: Error during recording: {e}")

        self.frame_count += 1

        # Reset action to NONE if no new key pressed (for continuous recording)
        # self.last_action = "NONE"  # Uncomment if you want action to reset each frame

    def _MainScene__check_game_status(self):
        """Override game status check to auto-exit and save recording when game ends"""
        # Check if won (all seeds eaten)
        if self._MainScene__seeds.is_field_empty():
            self._save_and_exit_game(won=True)
            return

        # Check if lost (no lives remaining and death animation finished)
        if self.pacman.death_is_finished() and not SoundController.is_busy(SoundCh.PLAYER):
            if self.hp:
                # Still has lives, call parent's setup to continue
                self.setup()
                return
            # No lives left - save and exit
            self._save_and_exit_game(won=False)

    def _save_and_exit_game(self, won: bool):
        """Save recording and exit the game"""
        # Stop recording and save
        final_score = int(self._MainScene__score)
        lives_remaining = int(self.hp)

        self.recorder.stop_recording(final_score, lives_remaining)
        filepath = self.recorder.save_trajectory()

        if filepath:
            print("\n" + "=" * 70)
            print(f"GAME {'WON' if won else 'LOST'} - RECORDING SAVED")
            print("=" * 70)
            print(f"Saved to: {filepath}")
            stats = self.recorder.get_statistics()
            print(f"Frames recorded: {stats.get('total_frames', 0):,}")
            print(f"Duration: {stats.get('duration_seconds', 0):.1f}s")
            print(f"Final score: {final_score}")
            print(f"Lives remaining: {lives_remaining}")
            print("=" * 70 + "\n")

        # Exit the game
        import sys
        sys.exit(0)

    def on_last_exit(self) -> None:
        """Save recording when exiting"""
        super().on_last_exit()

        # Stop recording and save
        final_score = int(self._MainScene__score)
        lives_remaining = int(self.hp)

        self.recorder.stop_recording(final_score, lives_remaining)
        filepath = self.recorder.save_trajectory()

        if filepath:
            print("\n" + "=" * 70)
            print("RECORDING SAVED")
            print("=" * 70)
            print(f"Saved to: {filepath}")
            stats = self.recorder.get_statistics()
            print(f"Frames recorded: {stats.get('total_frames', 0):,}")
            print(f"Duration: {stats.get('duration_seconds', 0):.1f}s")
            print(f"Final score: {final_score}")
            print("=" * 70 + "\n")


class AIMainScene(MainScene):
    """
    Extended MainScene that uses AI to play

    This class replaces human input with model predictions
    """

    def __init__(self, map_color=None):
        super().__init__(map_color)

        # Load AI player
        from auto_play import AIPlayer
        self.ai_player = AIPlayer()

        self.frame_count = 0
        self.action_history = []  # Track last few actions for temporal features

        decisions_per_sec = config.GAME_FPS // config.INFERENCE_EVERY_N_FRAMES
        print("\n" + "=" * 70)
        print("AI MODE ENABLED")
        print("=" * 70)
        print("The trained model will control Pac-Man.")
        print("Watch how well it learned from your gameplay!")
        print(f"Inference frequency: {decisions_per_sec} decisions/second "
              f"(every {config.INFERENCE_EVERY_N_FRAMES} frame{'s' if config.INFERENCE_EVERY_N_FRAMES > 1 else ''})")
        print("=" * 70 + "\n")

    def get_game_state(self) -> dict:
        """Extract current game state (same as recording mode)"""
        pacman_pos = (self.pacman.rect.centerx // 8, self.pacman.rect.centery // 8)
        pacman_direction = self.pacman.rotate

        ghosts = []
        for ghost in self._MainScene__ghosts:
            ghost_pos = (ghost.rect.centerx // 8, ghost.rect.centery // 8)
            ghost_direction = getattr(ghost, 'rotate', None)

            if ghost.state == GhostStateEnum.CHASE:
                ghost_state = "CHASE"
            elif ghost.state == GhostStateEnum.SCATTER:
                ghost_state = "SCATTER"
            elif ghost.state == GhostStateEnum.FRIGHTENED:
                ghost_state = "FRIGHTENED"
            else:
                ghost_state = "CHASE"

            ghosts.append({
                "pos": ghost_pos,
                "state": ghost_state,
                "direction": ghost_direction
            })

        # Get collision map and seeds
        collision_map = self._MainScene__loader.collision_map
        seeds_map = self._MainScene__seeds._SeedContainer__seeds

        # Get remaining seed positions
        seed_positions = []
        for y in range(len(seeds_map)):
            for x in range(len(seeds_map[y])):
                if seeds_map[y][x]:
                    seed_positions.append((x, y))

        # Get energizer positions
        energizer_positions = [
            (energizer.x, energizer.y)
            for energizer in self._MainScene__seeds._SeedContainer__energizers
        ]

        # Find nearest items using BFS
        from feature_utils import find_nearest_item_bfs
        nearest_seed = find_nearest_item_bfs(pacman_pos, seed_positions, collision_map)
        nearest_energizer = find_nearest_item_bfs(pacman_pos, energizer_positions, collision_map) if energizer_positions else None

        # Get lives and score
        lives = int(self.hp)
        score = int(self._MainScene__score)

        return {
            "pacman_pos": pacman_pos,
            "pacman_direction": pacman_direction,
            "ghosts": ghosts,
            "collision_map": collision_map,
            "seeds_map": seeds_map,
            "seed_positions": seed_positions,
            "energizer_positions": energizer_positions,
            "nearest_seed": nearest_seed,
            "nearest_energizer": nearest_energizer,
            "lives": lives,
            "score": score,
            "action_history": self.action_history.copy()
        }

    def process_logic(self) -> None:
        """Override to inject AI actions"""

        # Get AI action at configured frequency
        # config.INFERENCE_EVERY_N_FRAMES controls responsiveness:
        #   1 = 60 predictions/sec (every frame, maximum responsiveness)
        #   2 = 30 predictions/sec (smoother, still very responsive)
        #   4 = 15 predictions/sec (previous default, more conservative)
        if self._MainScene__state == GameStateEnum.ACTION and self.frame_count % config.INFERENCE_EVERY_N_FRAMES == 0:
            try:
                # Get current state
                game_state = self.get_game_state()

                # Get AI action
                action_name = self.ai_player.get_action_name(game_state)

                # Inject action as keyboard event
                self._inject_ai_action(action_name)

                # DEBUG: Confirm action injection
                if config.DEBUG_MODE and self.frame_count % 60 == 0:
                    decisions_per_sec = config.GAME_FPS // config.INFERENCE_EVERY_N_FRAMES
                    print(f"[DEBUG] Frame {self.frame_count}: Injected action '{action_name}' "
                          f"(inferring at {decisions_per_sec} decisions/sec)")

            except Exception as e:
                print(f"Warning: Error during AI control: {e}")

        self.frame_count += 1

        # Call parent logic
        super().process_logic()

    def _inject_ai_action(self, action_name: str):
        """Inject AI action into the game"""
        # Map action to key press
        key_map = {
            "UP": pygame.K_UP,
            "DOWN": pygame.K_DOWN,
            "LEFT": pygame.K_LEFT,
            "RIGHT": pygame.K_RIGHT,
        }

        action_to_idx = {
            "UP": 0,
            "DOWN": 1,
            "LEFT": 2,
            "RIGHT": 3
        }

        if action_name in key_map:
            # Track action in history
            if action_name in action_to_idx:
                self.action_history.append(action_to_idx[action_name])
                # Keep only last 10 actions
                if len(self.action_history) > 10:
                    self.action_history = self.action_history[-10:]

            # Create fake key event with mod attribute (required by pygame)
            key_event = pygame.event.Event(pygame.KEYDOWN, key=key_map[action_name], mod=0)
            pygame.event.post(key_event)

    def on_last_exit(self) -> None:
        """Print AI statistics when exiting"""
        super().on_last_exit()

        print("\n" + "=" * 70)
        print("AI GAME FINISHED")
        print("=" * 70)
        print(f"Final score: {int(self._MainScene__score)}")
        print(f"Lives remaining: {int(self.hp)}")
        self.ai_player.print_statistics()


def run_game(mode: str = "human"):
    """
    Run the Pac-Man game in specified mode

    Args:
        mode: "human", "record", or "ai"
    """
    # Change to Pacman directory
    os.chdir(PACMAN_DIR)

    # Initialize pygame (required for font and other modules)
    pygame.init()
    pygame.font.init()

    # Create game instance
    game = Game()

    # Replace main scene based on mode
    if mode == "record":
        SceneManager().reset(RecordingMainScene())
    elif mode == "ai":
        # Check if model exists
        if not config.MODEL_SAVE_PATH.exists():
            print("\n" + "=" * 70)
            print("ERROR: No trained model found!")
            print("=" * 70)
            print(f"Expected path: {config.MODEL_SAVE_PATH}")
            print("\nPlease train a model first:")
            print("  python train.py")
            print("=" * 70 + "\n")
            return
        SceneManager().reset(AIMainScene())
    else:
        # Normal human play
        print("\n" + "=" * 70)
        print("NORMAL PLAY MODE")
        print("=" * 70)
        print("Playing Pac-Man in normal mode.")
        print("Use arrow keys or WASD to move.")
        print("=" * 70 + "\n")

    # Run game
    try:
        game.main_loop()
    except KeyboardInterrupt:
        print("\nGame interrupted by user")
    except Exception as e:
        print(f"\nError running game: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Pac-Man Behavior Cloning - Game Runner"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["human", "record", "ai"],
        default="human",
        help="Game mode: human (normal play), record (record gameplay), ai (AI plays)"
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("PAC-MAN BEHAVIOR CLONING - GAME RUNNER")
    print("=" * 70)
    print(f"Mode: {args.mode.upper()}")
    print("=" * 70)

    run_game(args.mode)


if __name__ == "__main__":
    main()
