import asyncio
import logging
import re
from typing import Optional, List, Tuple
from collections import Counter

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

    # inbox pentru fiecare agent
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

    # inițializeaza agentul
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
        self.reserved_tiles: Counter = Counter()

    def _current_pos(self, world: WorldState) -> Position:
        for a in world.agents:
            if a.agent_id == self.agent_id:
                return a.position
        return self._my_state().position

    # trimit operatia la mediu si astept raspuns
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

    # anunta toti agentii despre actiunea planificata
    async def _announce(self, action: str, details: str = ""):
        if action in ("PICK", "UNBLOCK_PICK"):
            match = re.search(r'\((\d+),(\d+)\)', details)
            if match:
                x, y = int(match.group(1)), int(match.group(2))
                self.reserved_tiles[Position(x, y)] += 1

        elif action in ("USE_TILE", "DROP"):
            self.reserved_tiles.clear()

        ts = self.env.elapsed_ms() / 1000
        content = {
            "action": action,
            "details": details,
            "agent_color": self.color,
            "timestamp": ts,
        }
        await self.bus.broadcast(self.agent_id, self.all_agent_ids, content)
        logging.info(
            f"[{ts:.3f}s][BROADCAST][{self.color.upper()} -> ALL] "
            f"{action}: {details}"
        )

    # verifica daca a venit mesaj
    async def _listen_messages(self):
        msg = await self.bus.receive(self.agent_id, timeout=0.01)
        if msg:
            sender_color = msg["content"].get("agent_color", f"Agent{msg['from']}")
            action = msg["content"].get("action", "?")
            details = msg["content"].get("details", "")
            ts = self.env.elapsed_ms() / 1000
            logging.info(
                f"[{ts:.3f}s][INBOX][{self.color.upper()} ← {sender_color.upper()}] "
                f"{action}: {details}"
            )
            if action in ("PICK", "UNBLOCK_PICK"):
                match = re.search(r'\((\d+),(\d+)\)', details)
                if match:
                    x, y = int(match.group(1)), int(match.group(2))
                    self.reserved_tiles[Position(x, y)] += 1
                    logging.info(
                        f"[{ts:.3f}s][RESERVE][{self.color.upper()}] "
                        f"dala la ({x},{y}) rezervata de {sender_color.upper()}"
                    )
            elif action in ("USE_TILE", "DROP"):
                self.reserved_tiles.clear()

    def _score_task(
        self,
        tile_pos: Position,
        hole_pos: Position,
        world: WorldState,
    ) -> float:

        blocked = self._other_agent_positions()
        my_pos = self._my_state().position

        # calc drumul pana la dala
        path_to_tile = find_path(my_pos, tile_pos, world, set())
        if path_to_tile is None:
            return float("-inf")

        # calc drumul de la dala la groapa
        nav = find_path_adjacent_to(tile_pos, hole_pos, world, set())
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

    # itereaza toate combinatiile posibile si le scoreaza
    def _best_task(
            self,
            world: WorldState,
    ) -> Optional[Tuple[Position, Position, float]]:
        own_tiles = [
            tp for tp, stacks in world.tiles.items()
            for s in stacks if s.color == self.color and s.count > 0 and self.reserved_tiles[tp] < s.count
        ]
        own_holes = [
            h.position for h in world.holes
            if h.color == self.color and not h.is_filled
        ]

        # Colectează pozițiile celorlalți agenți — doar cei care NU transportă nimic
        other_free_agents = [
            a for a in world.agents
            if a.agent_id != self.agent_id and a.carried_tile is None
        ]

        best_score = float("-inf")
        best_tile = None
        best_hole = None

        for tp in own_tiles:
            for hp in own_holes:
                my_path = find_path(self._current_pos(world), tp, world, set())
                my_dist = len(my_path) if my_path is not None else 999

                # Verifică doar agenții liberi (cei ocupați nu contează)
                skip = False
                for other_agent in other_free_agents:
                    their_path = find_path(other_agent.position, tp, world, set())
                    their_dist = len(their_path) if their_path is not None else 999
                    if their_dist < my_dist:
                        skip = True
                        break

                if skip:
                    ts = self.env.elapsed_ms() / 1000
                    logging.info(
                        f"[{ts:.3f}s][SKIP TASK][{self.color.upper()}] "
                        f"dala la ({tp.x},{tp.y}) — alt agent liber mai aproape, o las lui"
                    )
                    continue

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

    # executa lista de miscari si daca esueaza returneaza false pentru replanificare
    async def _follow_path(self, path: list[str]) -> bool:
        ts = self.env.elapsed_ms() / 1000
        logging.info(
            f"[{ts:.3f}s][MOVE][{self.color.upper()}] "
            f"urmez drumul: {' -> '.join(path)} ({len(path)} pasi)"
        )
        for direction in path:
            if not self.env.running:
                return False
            result = await self.move(direction)
            if not result.success:
                if not self.env.running:
                    return False
                ts = self.env.elapsed_ms() / 1000
                logging.warning(
                    f"[{ts:.3f}s][BLOCKED][{self.color.upper()}] "
                    f"miscare {direction} blocata: {result.message} — replanific"
                )
                return False
            ts = self.env.elapsed_ms() / 1000
            pos = self._my_state().position
            logging.info(
                f"[{ts:.3f}s][STEP][{self.color.upper()}] "
                f"mers {direction} → acum la ({pos.x},{pos.y})"
            )
        return True

    async def run(self):
        logging.info(
            f"\n{'=' * 55}\n"
            f"[START] Agent {self.agent_id} ({self.color.upper()}) pornit\n"
            f"{'=' * 55}"
        )

        while self.env.running:
            agent_state = self._my_state()
            if agent_state is None:
                break

            await self._drain_messages()

            ts = self.env.elapsed_ms() / 1000
            carried = agent_state.carried_tile if agent_state.carried_tile else "nimic"
            logging.info(
                f"[{ts:.3f}s][TICK][{self.color.upper()}] "
                f"pos=({agent_state.position.x},{agent_state.position.y}) | "
                f"transporta={carried} | "
                f"puncte={agent_state.points}"
            )

            world = self.env.world

            if agent_state.carried_tile is None:
                await self._phase_collect_tile(world, agent_state)
            else:
                await self._phase_deliver_tile(world, agent_state)

            await asyncio.sleep(0.01)

        final_state = self._my_state()
        final_pts = final_state.points if final_state else "?"
        logging.info(
            f"\n{'=' * 55}\n"
            f"[STOP] Agent {self.agent_id} ({self.color.upper()}) oprit — "
            f"puncte finale: {final_pts}\n"
            f"{'=' * 55}"
        )

    async def _phase_collect_tile(self, world: WorldState, agent_state):
        await self._drain_messages()
        ts = self.env.elapsed_ms() / 1000
        logging.info(
            f"[{ts:.3f}s][PLAN][{self.color.upper()}] "
            f"caut cel mai bun task (nu transport nimic)..."
        )

        best = self._best_task(world)

        if best is not None:
            tile_pos, hole_pos, score = best
            ts = self.env.elapsed_ms() / 1000
            logging.info(
                f"[{ts:.3f}s][TASK][{self.color.upper()}] "
                f"task ales: dala {self.color} la ({tile_pos.x},{tile_pos.y}) "
                f"-> groapa la ({hole_pos.x},{hole_pos.y}) | scor={score:.1f}"
            )
            await self._announce(
                "PICK",
                f"{self.color} tile at ({tile_pos.x},{tile_pos.y}) "
                f"for hole ({hole_pos.x},{hole_pos.y}) [score={score:.1f}]"
            )
            await self._go_pick_tile(world, agent_state, tile_pos, self.color)
            return

        ts = self.env.elapsed_ms() / 1000
        logging.info(
            f"[{ts:.3f}s][FALLBACK][{self.color.upper()}] "
            f"nu am task valid cu dale proprii — caut orice dala apropiata"
        )

        blocked = self._other_agent_positions()
        any_tile = self._nearest_any_tile(world, agent_state.position, blocked)
        if any_tile:
            tile_pos, tile_color = any_tile
            ts = self.env.elapsed_ms() / 1000
            logging.info(
                f"[{ts:.3f}s][FALLBACK][{self.color.upper()}] "
                f"ridic dala {tile_color} la ({tile_pos.x},{tile_pos.y}) pentru deblocare"
            )
            await self._announce(
                "UNBLOCK_PICK",
                f"{tile_color} tile at ({tile_pos.x},{tile_pos.y})"
            )
            await self._go_pick_tile(world, agent_state, tile_pos, tile_color)
        else:
            ts = self.env.elapsed_ms() / 1000
            logging.info(
                f"[{ts:.3f}s][WAIT][{self.color.upper()}] "
                f"nu exista nicio dala accesibila — astept..."
            )
            await asyncio.sleep(0.1)

    # cauta cea mai apropiata dala de orice cul accesibila de la poz actuala
    def _nearest_any_tile(
            self,
            world: WorldState,
            pos: Position,
            blocked: set,
    ) -> Optional[Tuple[Position, str]]:
        ts = self.env.elapsed_ms() / 1000
        logging.info(
            f"[{ts:.3f}s][SEARCH][{self.color.upper()}] "
            f"caut orice dala accesibila de la ({pos.x},{pos.y})..."
        )

        # Agenți liberi (nu transportă nimic), alții decât mine
        other_free_agents = [
            a for a in world.agents
            if a.agent_id != self.agent_id and a.carried_tile is None
        ]

        best = None
        best_len = float("inf")

        for tile_pos, stacks in world.tiles.items():
            if tile_pos in blocked:
                continue

            for stack in stacks:
                if stack.count <= 0:
                    continue
                if self.reserved_tiles[tile_pos] >= stack.count:
                    ts2 = self.env.elapsed_ms() / 1000
                    logging.info(
                        f"[{ts2:.3f}s][SEARCH][{self.color.upper()}] "
                        f"  dalǎ {stack.color} la ({tile_pos.x},{tile_pos.y}): "
                        f"SARIT (rezervata de alt agent)"
                    )
                    continue

                path = find_path(pos, tile_pos, world, set())
                ts2 = self.env.elapsed_ms() / 1000
                if path is None:
                    logging.info(
                        f"[{ts2:.3f}s][SEARCH][{self.color.upper()}] "
                        f"  dalǎ {stack.color} la ({tile_pos.x},{tile_pos.y}): "
                        f"INACCESIBIL (drum blocat)"
                    )
                    continue

                my_dist = len(path)

                # *** FIX: skip dacă un alt agent liber e mai aproape ***
                skip = False
                for other in other_free_agents:
                    their_path = find_path(other.position, tile_pos, world, set())
                    their_dist = len(their_path) if their_path is not None else 999
                    if their_dist < my_dist:
                        skip = True
                        break

                if skip:
                    logging.info(
                        f"[{ts2:.3f}s][SEARCH][{self.color.upper()}] "
                        f"  dalǎ {stack.color} la ({tile_pos.x},{tile_pos.y}): "
                        f"SARIT (alt agent mai aproape)"
                    )
                    continue

                logging.info(
                    f"[{ts2:.3f}s][SEARCH][{self.color.upper()}] "
                    f"  dala {stack.color} la ({tile_pos.x},{tile_pos.y}): "
                    f"accesibila în {my_dist} pași"
                )
                if my_dist < best_len:
                    best_len = my_dist
                    best = (tile_pos, stack.color)

        ts = self.env.elapsed_ms() / 1000
        if best is not None:
            tile_pos, color = best
            logging.info(
                f"[{ts:.3f}s][SEARCH][{self.color.upper()}] "
                f"cea mai apropiata dala: {color} la ({tile_pos.x},{tile_pos.y}) "
                f"în {best_len} pași"
            )
        else:
            logging.info(
                f"[{ts:.3f}s][SEARCH][{self.color.upper()}] "
                f"nicio dala accesibila gasita — toate drumurile sunt blocate"
            )

        return best

    async def _go_pick_tile(
            self,
            world: WorldState,
            agent_state,
            tile_pos: Position,
            color: str,
    ):
        ts = self.env.elapsed_ms() / 1000
        logging.info(
            f"[{ts:.3f}s][TARGET][{self.color.upper()}] "
            f"merg să ridic dala {color} de la ({tile_pos.x},{tile_pos.y})"
        )

        if tile_pos == agent_state.position:
            result = await self.pick(color)
            ts = self.env.elapsed_ms() / 1000
            if result.success:
                logging.info(
                    f"[{ts:.3f}s][PICK OK][{self.color.upper()}] "
                    f"am ridicat dala {color} de la ({tile_pos.x},{tile_pos.y})"
                )
                self.reserved_tiles[tile_pos] = max(0, self.reserved_tiles[tile_pos] - 1)
                await self._announce("PICK_DONE", f"{color} tile at ({tile_pos.x},{tile_pos.y})")
            else:
                logging.info(
                    f"[{ts:.3f}s][PICK FAIL][{self.color.upper()}] "
                    f"dala {color} la ({tile_pos.x},{tile_pos.y}) nu mai e acolo — replanific"
                )
                self.reserved_tiles[tile_pos] = max(0, self.reserved_tiles[tile_pos] - 1)
                await asyncio.sleep(0.05)
            return

        # blocked = self._other_agent_positions()
        path = find_path(agent_state.position, tile_pos, world, set())
        if path is None:
            ts = self.env.elapsed_ms() / 1000
            logging.info(
                f"[{ts:.3f}s][NO PATH][{self.color.upper()}] "
                f"nu exista drum la ({tile_pos.x},{tile_pos.y}) — astept"
            )
            await asyncio.sleep(0.1)
            return

        success = await self._follow_path(path)
        if success:
            result = await self.pick(color)
            ts = self.env.elapsed_ms() / 1000
            if result.success:
                logging.info(
                    f"[{ts:.3f}s][PICK OK][{self.color.upper()}] "
                    f"am ridicat dala {color} de la ({tile_pos.x},{tile_pos.y})"
                )
                self.reserved_tiles[tile_pos] = max(0, self.reserved_tiles[tile_pos] - 1)
                await self._announce("PICK_DONE", f"{color} tile at ({tile_pos.x},{tile_pos.y})")
            else:
                logging.info(
                    f"[{ts:.3f}s][PICK FAIL][{self.color.upper()}] "
                    f"dala {color} la ({tile_pos.x},{tile_pos.y}) luata de altcineva — replanific"
                )
                self.reserved_tiles[tile_pos] = max(0, self.reserved_tiles[tile_pos] - 1)

    async def _phase_deliver_tile(self, world: WorldState, agent_state):
        tile_color = agent_state.carried_tile
        ts = self.env.elapsed_ms() / 1000
        logging.info(
            f"[{ts:.3f}s][DELIVER][{self.color.upper()}] "
            f"transport dala {tile_color} — caut cea mai buna groapa"
        )

        hole_pos = best_hole_for_color(tile_color, world)

        if hole_pos is None:
            active = world.get_active_holes()
            if active:
                hole_pos = active[0].position
                ts = self.env.elapsed_ms() / 1000
                logging.info(
                    f"[{ts:.3f}s][FALLBACK][{self.color.upper()}] "
                    f"nu am groapa de culoarea {tile_color} — merg la prima groapa activa ({hole_pos.x},{hole_pos.y})"
                )
            else:
                ts = self.env.elapsed_ms() / 1000
                logging.info(
                    f"[{ts:.3f}s][DROP][{self.color.upper()}] "
                    f"nu mai exista gropi active — las dala {tile_color} jos"
                )
                await self._announce("DROP", f"{tile_color} tile (no holes left)")
                await self.drop_tile()
                return
        else:
            ts = self.env.elapsed_ms() / 1000
            logging.info(
                f"[{ts:.3f}s][TARGET][{self.color.upper()}] "
                f"groapa tinta gasita la ({hole_pos.x},{hole_pos.y}) pentru dala {tile_color}"
            )

        await self._announce(
            "USE_TILE",
            f"{tile_color} tile -> hole at ({hole_pos.x},{hole_pos.y})"
        )

        # blocked = self._other_agent_positions()
        nav = find_path_adjacent_to(agent_state.position, hole_pos, world, set())

        if nav is None:
            ts = self.env.elapsed_ms() / 1000
            logging.info(
                f"[{ts:.3f}s][REROUTE][{self.color.upper()}] "
                f"groapa ({hole_pos.x},{hole_pos.y}) inaccesibila — caut alternativa"
            )
            for hole in world.get_active_holes():
                if hole.position == hole_pos:
                    continue
                nav = find_path_adjacent_to(
                    agent_state.position, hole.position, world, set()
                )
                if nav:
                    hole_pos = hole.position
                    ts = self.env.elapsed_ms() / 1000
                    logging.info(
                        f"[{ts:.3f}s][REROUTE][{self.color.upper()}] "
                        f"am găsit alternativa: groapa la ({hole_pos.x},{hole_pos.y})"
                    )
                    break

        if nav is None:
            ts = self.env.elapsed_ms() / 1000
            logging.info(
                f"[{ts:.3f}s][DROP][{self.color.upper()}] "
                f"nicio groapa accesibila — las dala jos și replanific"
            )
            await self.drop_tile()
            return

        move_path, use_direction = nav
        success = await self._follow_path(move_path)
        if not success:
            return

        result = await self.use_tile(use_direction)
        ts = self.env.elapsed_ms() / 1000
        if result.success:
            logging.info(
                f"[{ts:.3f}s][FILL OK][{self.color.upper()}] "
                f"am umplut groapa la ({hole_pos.x},{hole_pos.y}) cu dala {tile_color} "
                f"— directie: {use_direction}"
            )
        else:
            logging.info(
                f"[{ts:.3f}s][FILL FAIL][{self.color.upper()}] "
                f"groapa ({hole_pos.x},{hole_pos.y}) umpluta de altcineva — las dala jos"
            )
            await self.drop_tile()

    async def _drain_messages(self):
        """Procesează toate mesajele din inbox, nu doar unul."""
        while True:
            msg = await self.bus.receive(self.agent_id, timeout=0.0)
            if msg is None:
                break
            sender_color = msg["content"].get("agent_color", f"Agent{msg['from']}")
            action = msg["content"].get("action", "?")
            details = msg["content"].get("details", "")
            ts = self.env.elapsed_ms() / 1000
            logging.info(
                f"[{ts:.3f}s][INBOX][{self.color.upper()} ← {sender_color.upper()}] "
                f"{action}: {details}"
            )
            if action in ("PICK", "UNBLOCK_PICK"):
                match = re.search(r'\((\d+),(\d+)\)', details)
                if match:
                    x, y = int(match.group(1)), int(match.group(2))
                    self.reserved_tiles[Position(x, y)] += 1
                    logging.info(
                        f"[{ts:.3f}s][RESERVE][{self.color.upper()}] "
                        f"dala la ({x},{y}) rezervata de {sender_color.upper()}"
                    )
            # Cu:
            elif action == "PICK_DONE":
                match = re.search(r'\((\d+),(\d+)\)', details)
                if match:
                    x, y = int(match.group(1)), int(match.group(2))
                    pos = Position(x, y)
                    self.reserved_tiles[pos] = max(0, self.reserved_tiles[pos] - 1)
                    logging.info(
                        f"[{ts:.3f}s][RELEASE][{self.color.upper()}] "
                        f"dala la ({x},{y}) eliberata (ridicata de {sender_color.upper()})"
                    )
            elif action in ("USE_TILE", "DROP"):
                pass