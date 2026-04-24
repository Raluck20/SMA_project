from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from models import Position, SimulationConfig


@dataclass
class TileStack:
    """Dale de aceeași culoare într-o celulă."""
    color: str
    count: int


@dataclass
class HoleState:
    color: str
    depth: int       # scade cu 1 la fiecare dală plasată
    position: Position

    @property
    def is_filled(self) -> bool:
        return self.depth <= 0


@dataclass
class AgentState:
    agent_id: int
    color: str
    position: Position
    points: int = 0
    carried_tile: Optional[str] = None   # culoarea dalei transportate, sau None


@dataclass
class WorldState:
    """Starea completă a simulării la un moment dat."""
    width: int
    height: int
    obstacles: Set[Position]

    # Fiecare celulă poate avea mai multe stive de dale (una per culoare)
    tiles: Dict[Position, List[TileStack]] = field(default_factory=dict)

    holes: List[HoleState] = field(default_factory=list)
    agents: List[AgentState] = field(default_factory=list)

    def get_agent(self, agent_id: int) -> Optional[AgentState]:
        for a in self.agents:
            if a.agent_id == agent_id:
                return a
        return None

    def get_tiles_at(self, pos: Position) -> List[TileStack]:
        return self.tiles.get(pos, [])

    def get_tile_color_at(self, pos: Position, color: str) -> Optional[TileStack]:
        for stack in self.get_tiles_at(pos):
            if stack.color == color:
                return stack
        return None

    def get_holes_at(self, pos: Position) -> List[HoleState]:
        return [h for h in self.holes if h.position == pos]

    def get_active_holes(self) -> List[HoleState]:
        return [h for h in self.holes if not h.is_filled]

    def is_passable(self, pos: Position) -> bool:
        """O celulă e traversabilă dacă nu e obstacol și nu e groapă neacoperită."""
        if pos in self.obstacles:
            return False
        if not (0 <= pos.x < self.width and 0 <= pos.y < self.height):
            return False
        for hole in self.holes:
            if hole.position == pos and not hole.is_filled:
                return False
        return True

    def is_adjacent(self, a: Position, b: Position) -> bool:
        return abs(a.x - b.x) + abs(a.y - b.y) == 1

    def add_tile(self, pos: Position, color: str, count: int = 1):
        if pos not in self.tiles:
            self.tiles[pos] = []
        for stack in self.tiles[pos]:
            if stack.color == color:
                stack.count += count
                return
        self.tiles[pos].append(TileStack(color=color, count=count))

    def remove_tile(self, pos: Position, color: str) -> bool:
        """Elimină o dală de culoarea specificată. Returnează True dacă a reușit."""
        for stack in self.get_tiles_at(pos):
            if stack.color == color and stack.count > 0:
                stack.count -= 1
                if stack.count == 0:
                    self.tiles[pos].remove(stack)
                    if not self.tiles[pos]:
                        del self.tiles[pos]
                return True
        return False

    def snapshot(self) -> dict:
        """Returnează un snapshot al stării pentru Request_state() al agenților."""
        return {
            "obstacles": list(self.obstacles),
            "tiles": {
                str(pos): [{"color": s.color, "count": s.count} for s in stacks]
                for pos, stacks in self.tiles.items()
            },
            "holes": [
                {"color": h.color, "depth": h.depth, "position": (h.position.x, h.position.y)}
                for h in self.holes
            ],
            "agents": [
                {
                    "id": a.agent_id,
                    "color": a.color,
                    "position": (a.position.x, a.position.y),
                    "points": a.points,
                    "carried_tile": a.carried_tile
                }
                for a in self.agents
            ]
        }


def build_world(config: SimulationConfig) -> WorldState:
    world = WorldState(
        width=config.width,
        height=config.height,
        obstacles=set(config.obstacles),
    )

    for tg in config.tile_groups:
        world.add_tile(tg.position, tg.color, tg.count)

    for hc in config.holes:
        world.holes.append(HoleState(
            color=hc.color,
            depth=hc.depth,
            position=hc.position,
        ))

    for ac in config.agents:
        world.agents.append(AgentState(
            agent_id=ac.agent_id,
            color=ac.color,
            position=ac.position,
        ))

    return world