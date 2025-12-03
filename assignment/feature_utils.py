"""
Feature Extraction Utilities for Pac-Man Behavior Cloning

This module provides helper functions for extracting sophisticated features
from the game state to improve neural network learning.
"""

from typing import Tuple, List, Optional, Set
from collections import deque
import numpy as np


def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    """
    Calculate Manhattan distance between two positions.

    Args:
        pos1: (x, y) tuple
        pos2: (x, y) tuple

    Returns:
        Manhattan distance (int)
    """
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def euclidean_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    """
    Calculate Euclidean distance between two positions.

    Args:
        pos1: (x, y) tuple
        pos2: (x, y) tuple

    Returns:
        Euclidean distance (float)
    """
    dx = pos1[0] - pos2[0]
    dy = pos1[1] - pos2[1]
    return (dx * dx + dy * dy) ** 0.5


def find_nearest_item_bfs(
    start_pos: Tuple[int, int],
    items: List[Tuple[int, int]],
    collision_map: List[List[bool]],
    max_depth: int = 50
) -> Optional[Tuple[int, int]]:
    """
    Find nearest item using BFS pathfinding that respects walls.

    Args:
        start_pos: Starting position (x, y)
        items: List of item positions [(x, y), ...]
        collision_map: 2D boolean array (False = wall, True = passable)
        max_depth: Maximum search depth (default: 50)

    Returns:
        Position of nearest reachable item, or None if none found
    """
    if not items:
        return None

    # Convert items to set for O(1) lookup
    item_set = set(items)

    # BFS setup
    queue = deque([(start_pos, 0)])  # (position, distance)
    visited = {start_pos}

    height = len(collision_map)
    width = len(collision_map[0]) if height > 0 else 0

    # Direction offsets: up, down, left, right
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

    while queue:
        (x, y), dist = queue.popleft()

        # Check if we found an item
        if (x, y) in item_set:
            return (x, y)

        # Stop if we've searched too far
        if dist >= max_depth:
            continue

        # Explore neighbors
        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            # Check bounds
            if not (0 <= nx < width and 0 <= ny < height):
                continue

            # Check if already visited
            if (nx, ny) in visited:
                continue

            # Check if passable (collision_map[y][x] format)
            if not collision_map[ny][nx]:
                continue  # Hit a wall

            visited.add((nx, ny))
            queue.append(((nx, ny), dist + 1))

    # No item found - return closest by Euclidean distance
    if items:
        return min(items, key=lambda item: euclidean_distance(start_pos, item))
    return None


def get_wall_adjacency(
    pos: Tuple[int, int],
    collision_map: List[List[bool]]
) -> Tuple[bool, bool, bool, bool]:
    """
    Check for walls adjacent to position in 4 directions.

    Args:
        pos: Position (x, y) to check
        collision_map: 2D boolean array (False = wall, True = passable)

    Returns:
        Tuple of (wall_up, wall_down, wall_left, wall_right)
        True means there IS a wall in that direction
    """
    x, y = pos
    height = len(collision_map)
    width = len(collision_map[0]) if height > 0 else 0

    # Check each direction (out of bounds counts as wall)
    wall_up = (y - 1 < 0) or (not collision_map[y - 1][x])
    wall_down = (y + 1 >= height) or (not collision_map[y + 1][x])
    wall_left = (x - 1 < 0) or (not collision_map[y][x - 1])
    wall_right = (x + 1 >= width) or (not collision_map[y][x + 1])

    return (wall_up, wall_down, wall_left, wall_right)


def get_valid_moves(
    pos: Tuple[int, int],
    collision_map: List[List[bool]]
) -> Tuple[bool, bool, bool, bool]:
    """
    Check which directions are valid (no wall) from current position.

    Args:
        pos: Position (x, y) to check
        collision_map: 2D boolean array (False = wall, True = passable)

    Returns:
        Tuple of (can_go_up, can_go_down, can_go_left, can_go_right)
        True means that direction is passable
    """
    wall_up, wall_down, wall_left, wall_right = get_wall_adjacency(pos, collision_map)
    return (not wall_up, not wall_down, not wall_left, not wall_right)


def calculate_seed_density(
    pos: Tuple[int, int],
    seeds_map: List[List[bool]],
    radius: int = 5
) -> float:
    """
    Calculate local seed density around position.

    Args:
        pos: Center position (x, y)
        seeds_map: 2D boolean array of seed locations
        radius: Radius to check (default: 5)

    Returns:
        Ratio of seeds to total cells in the area (0.0 to 1.0)
    """
    x, y = pos
    height = len(seeds_map)
    width = len(seeds_map[0]) if height > 0 else 0

    seed_count = 0
    total_cells = 0

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            nx, ny = x + dx, y + dy

            # Check bounds
            if 0 <= nx < width and 0 <= ny < height:
                total_cells += 1
                if seeds_map[ny][nx]:
                    seed_count += 1

    return seed_count / max(total_cells, 1)


def predict_ghost_position(
    ghost_pos: Tuple[int, int],
    ghost_direction: Optional[int],
    steps: int = 2
) -> Tuple[int, int]:
    """
    Predict future ghost position assuming straight-line movement.

    Args:
        ghost_pos: Current ghost position (x, y)
        ghost_direction: Ghost direction (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, None=stationary)
        steps: Number of steps to predict ahead

    Returns:
        Predicted position (x, y)
    """
    if ghost_direction is None:
        return ghost_pos

    x, y = ghost_pos

    # Direction offsets
    direction_offsets = {
        0: (0, -steps),  # UP
        1: (0, steps),   # DOWN
        2: (-steps, 0),  # LEFT
        3: (steps, 0)    # RIGHT
    }

    dx, dy = direction_offsets.get(ghost_direction, (0, 0))
    return (x + dx, y + dy)


def detect_trap_situation(
    pacman_pos: Tuple[int, int],
    ghost_positions: List[Tuple[int, int]],
    collision_map: List[List[bool]],
    danger_radius: int = 3
) -> Tuple[float, int]:
    """
    Detect if Pac-Man is in a dangerous/trapped situation.

    Args:
        pacman_pos: Pac-Man's position (x, y)
        ghost_positions: List of ghost positions [(x, y), ...]
        collision_map: 2D boolean array (False = wall, True = passable)
        danger_radius: Distance threshold for "nearby" ghosts

    Returns:
        Tuple of (danger_score, num_escape_routes)
        - danger_score: 0.0 (safe) to 1.0 (extremely dangerous)
        - num_escape_routes: Number of directions leading away from danger
    """
    # Count valid moves
    can_up, can_down, can_left, can_right = get_valid_moves(pacman_pos, collision_map)
    valid_directions = sum([can_up, can_down, can_left, can_right])

    # Count nearby ghosts
    nearby_ghosts = sum(
        1 for ghost_pos in ghost_positions
        if manhattan_distance(pacman_pos, ghost_pos) <= danger_radius
    )

    # Calculate danger score
    # More nearby ghosts + fewer escape routes = higher danger
    if valid_directions == 0:
        danger_score = 1.0  # Completely trapped
    else:
        # Normalize: 0 ghosts = 0.0, 4 ghosts at close range = 1.0
        ghost_factor = min(nearby_ghosts / 4.0, 1.0)
        # Fewer escape routes increases danger
        escape_factor = 1.0 - (valid_directions / 4.0)
        danger_score = (ghost_factor + escape_factor) / 2.0

    # Count escape routes (directions away from nearest ghost)
    if not ghost_positions:
        num_escape_routes = valid_directions
    else:
        nearest_ghost = min(ghost_positions, key=lambda g: manhattan_distance(pacman_pos, g))
        escape_routes = 0

        # Check each valid direction
        x, y = pacman_pos
        gx, gy = nearest_ghost

        if can_up and y > gy:  # Moving away from ghost
            escape_routes += 1
        if can_down and y < gy:
            escape_routes += 1
        if can_left and x > gx:
            escape_routes += 1
        if can_right and x < gx:
            escape_routes += 1

        num_escape_routes = escape_routes

    return danger_score, num_escape_routes


def get_directional_distance(
    from_pos: Tuple[int, int],
    to_pos: Tuple[int, int]
) -> Tuple[float, float]:
    """
    Get normalized directional distance (like a vector).

    Args:
        from_pos: Source position (x, y)
        to_pos: Target position (x, y)

    Returns:
        Tuple of (normalized_dx, normalized_dy)
        Values are in range [-1, 1] where magnitude represents distance
    """
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]

    # Normalize by maximum possible distance (diagonal of grid)
    # Using approximate grid size (28x31 for standard Pac-Man)
    max_dist = (28**2 + 31**2) ** 0.5

    return (dx / max_dist, dy / max_dist)


def count_seeds_in_direction(
    pos: Tuple[int, int],
    direction: int,
    seeds_map: List[List[bool]],
    look_ahead: int = 5
) -> int:
    """
    Count seeds in a specific direction from position.

    Args:
        pos: Starting position (x, y)
        direction: Direction to look (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)
        seeds_map: 2D boolean array of seed locations
        look_ahead: How many cells to check ahead

    Returns:
        Number of seeds found in that direction
    """
    x, y = pos
    height = len(seeds_map)
    width = len(seeds_map[0]) if height > 0 else 0

    direction_offsets = {
        0: (0, -1),  # UP
        1: (0, 1),   # DOWN
        2: (-1, 0),  # LEFT
        3: (1, 0)    # RIGHT
    }

    dx, dy = direction_offsets.get(direction, (0, 0))
    seed_count = 0

    for step in range(1, look_ahead + 1):
        nx, ny = x + dx * step, y + dy * step

        if 0 <= nx < width and 0 <= ny < height:
            if seeds_map[ny][nx]:
                seed_count += 1

    return seed_count


def encode_action_history(
    action_history: List[int],
    num_actions: int = 5
) -> np.ndarray:
    """
    Encode recent action history as one-hot vectors.

    Args:
        action_history: List of recent action indices (most recent last)
        num_actions: Total number of possible actions (default: 5 for UP/DOWN/LEFT/RIGHT/NONE)

    Returns:
        Flattened one-hot encoded history array
    """
    # Take the last few actions (pad with 0 if not enough history)
    history_length = 3  # Keep last 3 actions
    recent_actions = action_history[-history_length:] if action_history else []

    # Pad with "no action" (0) if not enough history
    while len(recent_actions) < history_length:
        recent_actions.insert(0, 0)

    # One-hot encode each action
    encoded = np.zeros((history_length, num_actions))
    for i, action in enumerate(recent_actions):
        if 0 <= action < num_actions:
            encoded[i, action] = 1

    return encoded.flatten()


def get_ghost_relative_direction(
    pacman_pos: Tuple[int, int],
    ghost_pos: Tuple[int, int]
) -> Tuple[float, float, float, float]:
    """
    Get one-hot-like representation of ghost's direction relative to Pac-Man.

    Args:
        pacman_pos: Pac-Man position (x, y)
        ghost_pos: Ghost position (x, y)

    Returns:
        Tuple of (is_above, is_below, is_left, is_right)
        Values are 0.0 or 1.0 based on relative position
    """
    px, py = pacman_pos
    gx, gy = ghost_pos

    is_above = 1.0 if gy < py else 0.0
    is_below = 1.0 if gy > py else 0.0
    is_left = 1.0 if gx < px else 0.0
    is_right = 1.0 if gx > px else 0.0

    return (is_above, is_below, is_left, is_right)
