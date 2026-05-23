from pathlib import Path
from typing import List

from models import Position, AgentConfig, TileGroupConfig, HoleConfig, SimulationConfig


class ParseError(ValueError):
    pass


def _expect(tokens: List[str], index: int, expected: str) -> int:
    if index >= len(tokens) or tokens[index] != expected:
        got = tokens[index] if index < len(tokens) else "<EOF>"
        raise ParseError(f"Expected '{expected}', got '{got}' at token index {index}")
    return index + 1


def _read_int(tokens: List[str], index: int) -> tuple[int, int]:
    if index >= len(tokens):
        raise ParseError(f"Expected integer at token index {index}, got <EOF>")
    try:
        return int(tokens[index]), index + 1
    except ValueError as exc:
        raise ParseError(f"Expected integer at token index {index}, got '{tokens[index]}'") from exc


def load_system(path: str = "system.txt") -> SimulationConfig:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    tokens = file_path.read_text(encoding="utf-8").split()

    if len(tokens) < 5:
        raise ParseError("Input too short: expected at least 5 header values")

    idx = 0

    num_agents, idx = _read_int(tokens, idx)
    op_delay_ms, idx = _read_int(tokens, idx)
    total_time_ms, idx = _read_int(tokens, idx)
    width, idx = _read_int(tokens, idx)
    height, idx = _read_int(tokens, idx)

    if num_agents <= 0:
        raise ParseError("Number of agents must be positive")
    if op_delay_ms < 0 or total_time_ms <= 0:
        raise ParseError("Times must be valid positive integers")
    if width <= 0 or height <= 0:
        raise ParseError("Grid dimensions must be positive")

    colors = []
    for _ in range(num_agents):
        if idx >= len(tokens):
            raise ParseError("Not enough agent colors in input")
        colors.append(tokens[idx])
        idx += 1

    positions = []
    for _ in range(num_agents):
        x, idx = _read_int(tokens, idx)
        y, idx = _read_int(tokens, idx)
        positions.append(Position(x, y))

    agents = [
        AgentConfig(agent_id=i, color=colors[i], position=positions[i])
        for i in range(num_agents)
    ]

    idx = _expect(tokens, idx, "OBSTACLES")

    obstacles = set()
    while idx < len(tokens) and tokens[idx] != "TILES":
        x, idx = _read_int(tokens, idx)
        y, idx = _read_int(tokens, idx)
        obstacles.add(Position(x, y))

    idx = _expect(tokens, idx, "TILES")

    tile_groups = []
    while idx < len(tokens) and tokens[idx] != "HOLES":
        count, idx = _read_int(tokens, idx)
        if idx >= len(tokens):
            raise ParseError("Missing tile color")
        color = tokens[idx]
        idx += 1
        x, idx = _read_int(tokens, idx)
        y, idx = _read_int(tokens, idx)

        if count <= 0:
            raise ParseError("Tile group count must be positive")

        tile_groups.append(TileGroupConfig(count=count, color=color, position=Position(x, y)))

    idx = _expect(tokens, idx, "HOLES")

    holes = []
    while idx < len(tokens):
        depth, idx = _read_int(tokens, idx)
        if idx >= len(tokens):
            raise ParseError("Missing hole color")
        color = tokens[idx]
        idx += 1
        x, idx = _read_int(tokens, idx)
        y, idx = _read_int(tokens, idx)

        if depth <= 0:
            raise ParseError("Hole depth must be positive")

        holes.append(HoleConfig(depth=depth, color=color, position=Position(x, y)))

    config = SimulationConfig(
        num_agents=num_agents,
        operation_delay_ms=op_delay_ms,
        total_time_ms=total_time_ms,
        width=width,
        height=height,
        agents=agents,
        obstacles=obstacles,
        tile_groups=tile_groups,
        holes=holes,
    )

    _validate_config(config)
    return config


def _validate_config(config: SimulationConfig) -> None:
    def inside(pos: Position) -> bool:
        return 0 <= pos.x < config.width and 0 <= pos.y < config.height

    seen_agent_positions = set()

    for agent in config.agents:
        if not inside(agent.position):
            raise ParseError(f"Agent {agent.agent_id} out of bounds at {agent.position}")
        seen_agent_positions.add(agent.position)

    for obstacle in config.obstacles:
        if not inside(obstacle):
            raise ParseError(f"Obstacle out of bounds at {obstacle}")

    for tg in config.tile_groups:
        if not inside(tg.position):
            raise ParseError(f"Tile group out of bounds at {tg.position}")
        if tg.position in config.obstacles:
            raise ParseError(f"Tile group placed on obstacle at {tg.position}")

    for hole in config.holes:
        if not inside(hole.position):
            raise ParseError(f"Hole out of bounds at {hole.position}")
        if hole.position in config.obstacles:
            raise ParseError(f"Hole placed on obstacle at {hole.position}")

    for agent in config.agents:
        if agent.position in config.obstacles:
            raise ParseError(f"Agent {agent.agent_id} starts on obstacle at {agent.position}")