from dataclasses import dataclass, field
from typing import List, Set


@dataclass(frozen=True)
class Position:
    x: int
    y: int


@dataclass
class AgentConfig:
    agent_id: int
    color: str
    position: Position


@dataclass
class TileGroupConfig:
    count: int
    color: str
    position: Position


@dataclass
class HoleConfig:
    depth: int
    color: str
    position: Position


@dataclass
class SimulationConfig:
    num_agents: int
    operation_delay_ms: int
    total_time_ms: int
    width: int
    height: int
    agents: List[AgentConfig] = field(default_factory=list)
    obstacles: Set[Position] = field(default_factory=set)
    tile_groups: List[TileGroupConfig] = field(default_factory=list)
    holes: List[HoleConfig] = field(default_factory=list)