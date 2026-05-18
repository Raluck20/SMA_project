import asyncio
import time
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, Any

from models import Position, SimulationConfig
from world_state import WorldState, build_world


class OpType(Enum):
    MOVE = auto()
    PICK = auto()
    DROP_TILE = auto()
    USE_TILE = auto()
    TRANSFER_POINTS = auto()
    REQUEST_STATE = auto()


DIRECTION_DELTA = {
    "north": (0, 1),
    "south": (0, -1),
    "east":  (1, 0),
    "west":  (-1, 0),
}


@dataclass
class Operation:
    agent_id: int
    op_type: OpType
    args: dict
    reply_queue: asyncio.Queue


@dataclass
class OpResult:
    success: bool
    message: str
    data: Any = None


class Environment:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.world: WorldState = build_world(config)
        self.op_queue: asyncio.Queue = asyncio.Queue()
        self.start_time: Optional[float] = None
        self.running = False

        self.operation_log: list = []

        # tin minte daca ultima op a unui agent a fost req state pt ca nu pot face 2 la rand
        self._last_op_was_request: dict[int, bool] = {
            a.agent_id: False for a in config.agents
        }

    def elapsed_ms(self) -> float:
        if self.start_time is None:
            return 0.0
        return (time.time() - self.start_time) * 1000

    def log(self, msg: str):
        ts = self.elapsed_ms() / 1000
        entry = f"[{ts:.3f}][ENV] {msg}"
        self.operation_log.append(entry)

    async def run(self):
        self.start_time = time.time()
        self.running = True
        total_ms = self.config.total_time_ms

        # ruleaza pana la expirarea timpului
        while self.running:
            elapsed = self.elapsed_ms()
            if elapsed >= total_ms:
                self.running = False
                break

            if not self.world.get_active_holes():
                self.running = False
                break

            # astept max 50 ms o operatie din coada
            try:
                op: Operation = await asyncio.wait_for(
                    self.op_queue.get(), timeout=0.05
                )
            except asyncio.TimeoutError:
                continue

            # execut operatia
            result = self._validate_and_execute(op)

            # daca op a reusit si nu cere starea lumii astept t ms
            if result.success and op.op_type != OpType.REQUEST_STATE:
                await asyncio.sleep(self.config.operation_delay_ms / 1000)

            # trimit raspunsul inapoi agentului
            await op.reply_queue.put(result)

        await self._drain_queue()

    # golesc coada de operatii la terminarea simularii ca sa nu se blocheze codul
    async def _drain_queue(self):
        await asyncio.sleep(0.1)
        while True:
            try:
                op: Operation = self.op_queue.get_nowait()
                await op.reply_queue.put(OpResult(False, "Simulation stopped"))
            except asyncio.QueueEmpty:
                break

    def _validate_and_execute(self, op: Operation) -> OpResult:
        agent = self.world.get_agent(op.agent_id)
        if agent is None:
            return OpResult(False, f"Agent {op.agent_id} not found")

        # nu se poate cere starea de doua ori la rand
        if op.op_type == OpType.REQUEST_STATE:
            if self._last_op_was_request.get(op.agent_id, False):
                return OpResult(False, "Cannot call Request_state twice in a row")
            self._last_op_was_request[op.agent_id] = True
            return OpResult(True, "State returned", data=self.world.snapshot())

        self._last_op_was_request[op.agent_id] = False

        if op.op_type == OpType.MOVE:
            return self._exec_move(agent, op.args)
        elif op.op_type == OpType.PICK:
            return self._exec_pick(agent, op.args)
        elif op.op_type == OpType.DROP_TILE:
            return self._exec_drop(agent)
        elif op.op_type == OpType.USE_TILE:
            return self._exec_use_tile(agent, op.args)
        elif op.op_type == OpType.TRANSFER_POINTS:
            return self._exec_transfer_points(agent, op.args)

        return OpResult(False, f"Unknown operation {op.op_type}")

    def _exec_move(self, agent, args) -> OpResult:
        direction = args.get("direction", "").lower()
        delta = DIRECTION_DELTA.get(direction)
        if delta is None:
            return OpResult(False, f"Invalid direction '{direction}'")

        # calc noua pozitie
        new_pos = Position(agent.position.x + delta[0], agent.position.y + delta[1])

        # verific daca celula e blocata sau inafara gridului
        if not self.world.is_passable(new_pos):
            self.log(f"[{agent.color.upper()}] Move {direction} FAILED (blocked)")
            return OpResult(False, f"Cannot move to {new_pos}: blocked or out of bounds")

        # actualizez pozitia
        agent.position = new_pos
        self.log(f"[{agent.color.upper()}] Move {direction.capitalize()}")
        return OpResult(True, "Moved")

    def _exec_pick(self, agent, args) -> OpResult:
        color = args.get("color", "")
        if agent.carried_tile is not None:
            return OpResult(False, "Already carrying a tile")

        if not self.world.remove_tile(agent.position, color):
            self.log(f"[{agent.color.upper()}] Pick {color} FAILED (no tile here)")
            return OpResult(False, f"No tile of color '{color}' at {agent.position}")

        agent.carried_tile = color
        self.log(f"[{agent.color.upper()}] Pick {color}")
        return OpResult(True, "Picked")

    def _exec_drop(self, agent) -> OpResult:
        if agent.carried_tile is None:
            return OpResult(False, "Not carrying any tile")

        self.world.add_tile(agent.position, agent.carried_tile)
        self.log(f"[{agent.color.upper()}] Drop tile ({agent.carried_tile})")
        agent.carried_tile = None
        return OpResult(True, "Dropped")

    def _exec_use_tile(self, agent, args) -> OpResult:
        direction = args.get("direction", "").lower()
        delta = DIRECTION_DELTA.get(direction)
        if delta is None:
            return OpResult(False, f"Invalid direction '{direction}'")

        if agent.carried_tile is None:
            return OpResult(False, "Not carrying a tile")

        # calc directia gropii
        hole_pos = Position(agent.position.x + delta[0], agent.position.y + delta[1])

        # verific ca groapa e neacoperita complet
        adjacent_holes = [
            h for h in self.world.holes
            if h.position == hole_pos and not h.is_filled
        ]

        if not adjacent_holes:
            self.log(f"[{agent.color.upper()}] Use tile {direction} FAILED (no hole there)")
            return OpResult(False, f"No active hole adjacent in direction '{direction}'")

        # acopar groapa
        hole = adjacent_holes[0]
        tile_color = agent.carried_tile
        agent.carried_tile = None
        hole.depth -= 1

        # punctaj
        points_earned = 0
        if tile_color == hole.color:
            points_earned = 10
            if hole.is_filled:
                points_earned += 40

        # punctajul merge la agentul de aceasi culoare cu groapa
        for a in self.world.agents:
            if a.color == hole.color:
                a.points += points_earned
                break

        self.log(
            f"[{agent.color.upper()}] Use tile {direction} on hole@{hole_pos} "
            f"(depth now {hole.depth}, +{points_earned} pts to {hole.color})"
        )
        return OpResult(True, f"Used tile, hole depth now {hole.depth}", data=points_earned)

    # transfer puncte de la un agent la altul
    # folosit in negociere — un agent da puncte unui alt agent pentru a indeplini o sarcina in locul sau
    def _exec_transfer_points(self, agent, args) -> OpResult:
        target_id = args.get("target_id")
        points = args.get("points", 0)

        if not isinstance(points, int) or points <= 0:
            return OpResult(False, "Points must be a positive integer")

        target = self.world.get_agent(target_id)
        if target is None:
            return OpResult(False, f"Target agent {target_id} not found")

        agent.points -= points
        target.points += points
        self.log(
            f"[{agent.color.upper()}] Transfer {points} pts -> Agent {target_id}"
        )
        return OpResult(True, f"Transferred {points} points")