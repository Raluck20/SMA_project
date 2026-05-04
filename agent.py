import asyncio
import logging
from typing import Optional, List, Tuple

from models import Position
from world_state import WorldState
from environment import Environment, Operation, OpType, OpResult
from pathfinder import (
    find_path,
    find_path_adjacent_to,
    best_hole_for_color,
    reachable_tiles_of_color,
)


class MessageBus:

    def __init__(self, n_agents: int):
        self.inboxes: dict[int, asyncio.Queue] = {
            i: asyncio.Queue() for i in range(n_agents)
        }

    async def send(self, sender_id: int, receiver_id: int, content: dict):
        await self.inboxes[receiver_id].put({
            "from": sender_id,
            "content": content,
        })

    async def broadcast(self, sender_id: int, all_ids: List[int], content: dict):
        for rid in all_ids:
            if rid != sender_id:
                await self.send(sender_id, rid, content)

    async def receive(self, agent_id: int, timeout: float = 0.05) -> Optional[dict]:
        try:
            return await asyncio.wait_for(
                self.inboxes[agent_id].get(), timeout=timeout
            )
        except asyncio.TimeoutError:
            return None


class TileWorldAgent:

    def __init__(
        self,
        agent_id: int,
        color: str,
        env: Environment,
        bus: MessageBus,
        all_agent_ids: List[int],
    ):
        self.agent_id = agent_id
        self.color = color
        self.env = env
        self.bus = bus
        self.all_agent_ids = all_agent_ids
        self.reply_queue: asyncio.Queue = asyncio.Queue()

    async def _send_op(self, op_type: OpType, args: dict = None) -> OpResult:
        op = Operation(
            agent_id=self.agent_id,
            op_type=op_type,
            args=args or {},
            reply_queue=self.reply_queue,
        )
        await self.env.op_queue.put(op)
        try:
            return await asyncio.wait_for(self.reply_queue.get(), timeout=3.0)
        except asyncio.TimeoutError:
            return OpResult(False, "Simulation stopped")

    async def move(self, direction: str) -> OpResult:
        return await self._send_op(OpType.MOVE, {"direction": direction})

    async def pick(self, color: str) -> OpResult:
        return await self._send_op(OpType.PICK, {"color": color})

    async def drop_tile(self) -> OpResult:
        return await self._send_op(OpType.DROP_TILE)

    async def use_tile(self, direction: str) -> OpResult:
        return await self._send_op(OpType.USE_TILE, {"direction": direction})

    async def transfer_points(self, target_id: int, points: int) -> OpResult:
        return await self._send_op(OpType.TRANSFER_POINTS, {
            "target_id": target_id, "points": points
        })

    async def request_state(self) -> Optional[dict]:
        result = await self._send_op(OpType.REQUEST_STATE)
        return result.data if result.success else None

    async def _announce(self, action: str, details: str = ""):

        ts = self.env.elapsed_ms() / 1000
        content = {
            "action": action,
            "details": details,
            "agent_color": self.color,
            "timestamp": ts,
        }
        await self.bus.broadcast(self.agent_id, self.all_agent_ids, content)
        logging.info(
            f"[{ts:.3f}][MSG][{self.color.upper()} -> ALL] "
            f"{action}: {details}"
        )

    async def _listen_messages(self):
        msg = await self.bus.receive(self.agent_id, timeout=0.01)
        if msg:
            sender_color = msg["content"].get("agent_color", f"Agent{msg['from']}")
            action = msg["content"].get("action", "?")
            details = msg["content"].get("details", "")
            ts = self.env.elapsed_ms() / 1000
            logging.info(
                f"[{ts:.3f}][MSG][{self.color.upper()} received from {sender_color.upper()}] "
                f"{action}: {details}"
            )

    def _score_task(
        self,
        tile_pos: Position,
        hole_pos: Position,
        world: WorldState,
    ) -> float:

        blocked = self._other_agent_positions()
        my_pos = self._my_state().position

        path_to_tile = find_path(my_pos, tile_pos, world, blocked)
        if path_to_tile is None:
            return float("-inf")

        nav = find_path_adjacent_to(tile_pos, hole_pos, world, blocked)
        if nav is None:
            return float("-inf")

        total_steps = len(path_to_tile) + len(nav[0])

        hole = next((h for h in world.holes if h.position == hole_pos), None)
        if hole is None or hole.is_filled:
            return float("-inf")

        base_reward = 10
        bonus = 40 if hole.depth == 1 else 0
        reward = base_reward + bonus

        step_penalty = 0.5
        return reward - total_steps * step_penalty

    def _best_task(
        self,
        world: WorldState,
    ) -> Optional[Tuple[Position, Position, float]]:
        own_tiles = [
            tp for tp, stacks in world.tiles.items()
            for s in stacks if s.color == self.color and s.count > 0
        ]
        own_holes = [
            h.position for h in world.holes
            if h.color == self.color and not h.is_filled
        ]

        best_score = float("-inf")
        best_tile = None
        best_hole = None

        for tp in own_tiles:
            for hp in own_holes:
                score = self._score_task(tp, hp, world)
                if score > best_score:
                    best_score = score
                    best_tile = tp
                    best_hole = hp

        if best_tile is None:
            return None
        return (best_tile, best_hole, best_score)

    def _my_state(self):
        return self.env.world.get_agent(self.agent_id)

    def _other_agent_positions(self) -> set:
        return {
            a.position for a in self.env.world.agents
            if a.agent_id != self.agent_id
        }

    async def _follow_path(self, path: list[str]) -> bool:
        for direction in path:
            if not self.env.running:
                return False
            result = await self.move(direction)
            if not result.success:
                if not self.env.running:
                    return False
                logging.warning(
                    f"Agent {self.agent_id} ({self.color}): "
                    f"move {direction} failed: {result.message}"
                )
                return False
        return True


    async def run(self):
        logging.info(f"Agent {self.agent_id} ({self.color}): started")

        while self.env.running:
            agent_state = self._my_state()
            if agent_state is None:
                break

            await self._listen_messages()

            world = self.env.world

            if agent_state.carried_tile is None:
                await self._phase_collect_tile(world, agent_state)
            else:
                await self._phase_deliver_tile(world, agent_state)

            await asyncio.sleep(0.01)

        final_pts = self._my_state().points if self._my_state() else "?"
        logging.info(
            f"Agent {self.agent_id} ({self.color}): stopped. "
            f"Final points: {final_pts}"
        )


    async def _phase_collect_tile(self, world: WorldState, agent_state):
        best = self._best_task(world)

        if best is not None:
            tile_pos, hole_pos, score = best
            logging.info(
                f"[{self.env.elapsed_ms()/1000:.3f}][PLAN] "
                f"Agent {self.agent_id} ({self.color}): "
                f"[SOLO] tile ({tile_pos.x},{tile_pos.y}) -> "
                f"hole ({hole_pos.x},{hole_pos.y}), score={score:.1f}"
            )
            await self._announce(
                "PICK",
                f"{self.color} tile at ({tile_pos.x},{tile_pos.y}) "
                f"for hole ({hole_pos.x},{hole_pos.y}) [score={score:.1f}]"
            )
            await self._go_pick_tile(world, agent_state, tile_pos, self.color)
            return

        blocked = self._other_agent_positions()
        any_tile = self._nearest_any_tile(world, agent_state.position, blocked)
        if any_tile:
            tile_pos, tile_color = any_tile
            await self._announce(
                "UNBLOCK_PICK",
                f"{tile_color} tile at ({tile_pos.x},{tile_pos.y})"
            )
            await self._go_pick_tile(world, agent_state, tile_pos, tile_color)
        else:
            await asyncio.sleep(0.1)

    def _nearest_any_tile(
        self,
        world: WorldState,
        pos: Position,
        blocked: set,
    ) -> Optional[Tuple[Position, str]]:
        best = None
        best_len = float("inf")
        for tile_pos, stacks in world.tiles.items():
            if tile_pos in blocked:
                continue
            for stack in stacks:
                if stack.count > 0:
                    path = find_path(pos, tile_pos, world, blocked)
                    if path is not None and len(path) < best_len:
                        best_len = len(path)
                        best = (tile_pos, stack.color)
        return best

    async def _go_pick_tile(
        self,
        world: WorldState,
        agent_state,
        tile_pos: Position,
        color: str,
    ):
        if tile_pos == agent_state.position:
            result = await self.pick(color)
            if result.success:
                logging.info(
                    f"Agent {self.agent_id} ({self.color}): "
                    f"picked {color} tile at ({tile_pos.x},{tile_pos.y})"
                )
            else:
                await asyncio.sleep(0.05)
            return

        blocked = self._other_agent_positions()
        path = find_path(agent_state.position, tile_pos, world, blocked)
        if path is None:
            await asyncio.sleep(0.1)
            return

        success = await self._follow_path(path)
        if success:
            result = await self.pick(color)
            if result.success:
                logging.info(
                    f"Agent {self.agent_id} ({self.color}): "
                    f"picked {color} tile at ({tile_pos.x},{tile_pos.y})"
                )
            else:
                logging.debug(
                    f"Agent {self.agent_id}: tile gone at {tile_pos}, retrying"
                )

    async def _phase_deliver_tile(self, world: WorldState, agent_state):
        tile_color = agent_state.carried_tile
        hole_pos = best_hole_for_color(tile_color, world)

        if hole_pos is None:
            active = world.get_active_holes()
            if active:
                hole_pos = active[0].position
            else:
                await self._announce("DROP", f"{tile_color} tile (no holes left)")
                await self.drop_tile()
                return

        await self._announce(
            "USE_TILE",
            f"{tile_color} tile -> hole at ({hole_pos.x},{hole_pos.y})"
        )

        blocked = self._other_agent_positions()
        nav = find_path_adjacent_to(agent_state.position, hole_pos, world, blocked)

        if nav is None:
            for hole in world.get_active_holes():
                if hole.position == hole_pos:
                    continue
                nav = find_path_adjacent_to(
                    agent_state.position, hole.position, world, blocked
                )
                if nav:
                    hole_pos = hole.position
                    break

        if nav is None:
            logging.debug(
                f"Agent {self.agent_id}: no path to any hole, dropping tile"
            )
            await self.drop_tile()
            return

        move_path, use_direction = nav
        success = await self._follow_path(move_path)
        if not success:
            return

        result = await self.use_tile(use_direction)
        if result.success:
            logging.info(
                f"Agent {self.agent_id} ({self.color}): "
                f"used {tile_color} tile on hole at ({hole_pos.x},{hole_pos.y})"
            )
        else:
            logging.debug(
                f"Agent {self.agent_id}: use_tile failed: {result.message}, dropping"
            )
            await self.drop_tile()