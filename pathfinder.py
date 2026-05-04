from collections import deque
from typing import List, Optional, Set, Tuple
from models import Position
from world_state import WorldState

DIRECTIONS = {
    "north": (0, 1),
    "south": (0, -1),
    "east":  (1, 0),
    "west":  (-1, 0),
}

OPPOSITE = {
    "north": "south",
    "south": "north",
    "east":  "west",
    "west":  "east",
}


def find_path(
    start: Position,
    goal: Position,
    world: WorldState,
    extra_blocked: Optional[Set[Position]] = None,
) -> Optional[List[str]]:

    if start == goal:
        return []

    blocked = extra_blocked or set()
    queue = deque([(start, [])])
    visited = {start}

    while queue:
        current, path = queue.popleft()

        for direction, (dx, dy) in DIRECTIONS.items():
            neighbor = Position(current.x + dx, current.y + dy)

            if neighbor in visited:
                continue
            if neighbor in blocked:
                continue

            # Permitem goal chiar dacă e impassable (groapă, obstacol adiacent)
            if neighbor != goal and not world.is_passable(neighbor):
                continue

            new_path = path + [direction]

            if neighbor == goal:
                return new_path

            visited.add(neighbor)
            queue.append((neighbor, new_path))

    return None


def find_path_adjacent_to(
    start: Position,
    target: Position,
    world: WorldState,
    extra_blocked: Optional[Set[Position]] = None,
) -> Optional[Tuple[List[str], str]]:

    best = None

    for direction, (dx, dy) in DIRECTIONS.items():
        adjacent = Position(target.x - dx, target.y - dy)

        if adjacent == start:
            return ([], direction)

        if not world.is_passable(adjacent):
            continue

        path = find_path(start, adjacent, world, extra_blocked)
        if path is None:
            continue

        if best is None or len(path) < len(best[0]):
            best = (path, direction)

    return best


def find_path_adjacent_to_any(
    start: Position,
    targets: List[Position],
    world: WorldState,
    extra_blocked: Optional[Set[Position]] = None,
) -> Optional[Tuple[List[str], str, Position]]:

    best = None

    for target in targets:
        result = find_path_adjacent_to(start, target, world, extra_blocked)
        if result is None:
            continue
        path, direction = result
        if best is None or len(path) < len(best[0]):
            best = (path, direction, target)

    return best

def manhattan(a: Position, b: Position) -> int:
    return abs(a.x - b.x) + abs(a.y - b.y)


def nearest_tile_of_color(
    pos: Position,
    color: str,
    world: WorldState,
    extra_blocked: Optional[Set[Position]] = None,
) -> Optional[Position]:

    blocked = extra_blocked or set()
    best_pos = None
    best_dist = float("inf")

    for tile_pos, stacks in world.tiles.items():
        if tile_pos in blocked:
            continue
        for stack in stacks:
            if stack.color == color and stack.count > 0:
                d = manhattan(pos, tile_pos)
                if d < best_dist:
                    best_dist = d
                    best_pos = tile_pos

    return best_pos

def best_hole_for_color(
    color: str,
    world: WorldState,
    prefer_depth_one: bool = True,
) -> Optional[Position]:

    holes = [h for h in world.holes if h.color == color and not h.is_filled]
    if not holes:
        return None

    if prefer_depth_one:
        depth_one = [h for h in holes if h.depth == 1]
        if depth_one:
            return depth_one[0].position

    return holes[0].position


def all_holes_for_color(
    color: str,
    world: WorldState,
) -> List[Position]:
    return [h.position for h in world.holes if h.color == color and not h.is_filled]


def reachable_tiles_of_color(
    pos: Position,
    color: str,
    world: WorldState,
    extra_blocked: Optional[Set[Position]] = None,
) -> List[Tuple[Position, int]]:

    blocked = extra_blocked or set()
    results = []

    for tile_pos, stacks in world.tiles.items():
        if tile_pos in blocked:
            continue
        for stack in stacks:
            if stack.color == color and stack.count > 0:
                path = find_path(pos, tile_pos, world, blocked)
                if path is not None:
                    results.append((tile_pos, len(path)))

    results.sort(key=lambda x: x[1])
    return results