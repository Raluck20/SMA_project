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
from dataclasses import dataclass, field
import uuid
import math
@dataclass
class ContractTask:
    task_id: str
    beneficiary_id: int
    beneficiary_color: str
    executor_id: Optional[int]
    tile_pos: Position
    tile_color: str
    hole_pos: Position
    hole_color: str
    agreed_price: int = 0
    status: str = "OPEN"
    proposals: dict = field(default_factory=dict)
    round_no: int = 0
    max_rounds: int = 3
    asking_price: int = 0

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

        self.open_contracts = {}
        self.accepted_contracts = {}
        self.current_contract = None
        self.contract_wait_until = 0
        self.pending_proposals = {}
        self.cfp_pending = {}
        self.forced_local_task = None
        self.negotiation_rounds_limit = 3

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
    
    def _estimate_reward(self, hole_pos: Position, world: WorldState) -> int:
        hole = next((h for h in world.holes if h.position == hole_pos), None)
        if hole is None or hole.is_filled:
            return 0
        return 10 + (40 if hole.depth == 1 else 0)

    def _estimate_steps(self, start: Position, tile_pos: Position, hole_pos: Position, world: WorldState) -> int:
        path1 = find_path(start, tile_pos, world, set())
        nav = find_path_adjacent_to(tile_pos, hole_pos, world, set())
        if path1 is None or nav is None:
            return 999
        return len(path1) + len(nav[0])

    def _estimate_own_task_value(self, tile_pos: Position, hole_pos: Position, world: WorldState) -> float:
        reward = self._estimate_reward(hole_pos, world)
        steps = self._estimate_steps(self._current_pos(world), tile_pos, hole_pos, world)
        return reward - 0.5 * steps
    
    def _make_task_id(self) -> str:
        return f"{self.agent_id}-{uuid.uuid4().hex[:8]}"
    
    async def _broadcast_cfp(self, task: ContractTask, expected_reward: int, asking_price: int):
        ts = self.env.elapsed_ms() / 1000
        content = {
            "action": "CFP",
            "task_id": task.task_id,
            "beneficiary_id": task.beneficiary_id,
            "tile_pos": (task.tile_pos.x, task.tile_pos.y),
            "tile_color": task.tile_color,
            "hole_pos": (task.hole_pos.x, task.hole_pos.y),
            "hole_color": task.hole_color,
            "expected_reward": expected_reward,
            "asking_price": asking_price,
            "round_no": task.round_no,
            "agent_color": self.color,
            "timestamp": ts,
        }
        await self.bus.broadcast(self.agent_id, self.all_agent_ids, content)

        logging.info(
            f"[{ts:.3f}s][CFP][{self.color.upper()} -> ALL] "
            f"task={task.task_id} round={task.round_no}/{task.max_rounds} "
            f"ask={asking_price} tile=({task.tile_pos.x},{task.tile_pos.y}) "
            f"hole=({task.hole_pos.x},{task.hole_pos.y}) reward={expected_reward}"
        )
    
    async def _send_proposal(self, receiver_id: int, task_id: str, price: int):
        ts = self.env.elapsed_ms() / 1000
        content = {
            "action": "PROPOSE",
            "task_id": task_id,
            "price": price,
            "agent_color": self.color,
            "timestamp": ts,
        }
        await self.bus.send(self.agent_id, receiver_id, content)
        logging.info(
            f"[{ts:.3f}s][PROPOSE][{self.color.upper()} -> {receiver_id}] "
            f"task={task_id} price={price}"
        )

    async def _send_accept(self, receiver_id: int, task: ContractTask):
        ts = self.env.elapsed_ms() / 1000
        content = {
            "action": "ACCEPT",
            "task_id": task.task_id,
            "beneficiary_id": task.beneficiary_id,
            "tile_pos": (task.tile_pos.x, task.tile_pos.y),
            "tile_color": task.tile_color,
            "hole_pos": (task.hole_pos.x, task.hole_pos.y),
            "hole_color": task.hole_color,
            "price": task.agreed_price,
            "agent_color": self.color,
            "timestamp": ts,
        }
        await self.bus.send(self.agent_id, receiver_id, content)
        logging.info(
            f"[{ts:.3f}s][ACCEPT][{self.color.upper()} -> {receiver_id}] "
            f"task={task.task_id} price={task.agreed_price}"
        )

    async def _send_task_done(self, receiver_id: int, task_id: str):
        ts = self.env.elapsed_ms() / 1000
        content = {
            "action": "TASK_DONE",
            "task_id": task_id,
            "agent_color": self.color,
            "timestamp": ts,
        }
        await self.bus.send(self.agent_id, receiver_id, content)
        logging.info(
            f"[{ts:.3f}s][TASK_DONE][{self.color.upper()} -> {receiver_id}] "
            f"task={task_id}"
        )

    def _choose_best_proposal(self, task_id: str):
        task = self.open_contracts.get(task_id)
        if task is None or not task.proposals:
            return None
        return min(task.proposals.items(), key=lambda kv: kv[1])  
        # returneaza (agent_id, price)

    async def _finalize_contracts(self):
        now = self.env.elapsed_ms() / 1000
        finished = []

        for task_id, wait_until in list(self.cfp_pending.items()):
            if now < wait_until:
                continue

            task = self.open_contracts.get(task_id)
            if task is None:
                finished.append(task_id)
                continue

            best = self._choose_best_proposal(task_id)

            if best is not None:
                winner_id, winner_price = best

                my_value = self._estimate_own_task_value(task.tile_pos, task.hole_pos, self.env.world)
                reward = self._estimate_reward(task.hole_pos, self.env.world)
                wait_penalty = 2.0 * task.round_no
                outsourced_value = reward - winner_price - wait_penalty
                if outsourced_value > my_value + 1:
                    task.executor_id = winner_id
                    task.agreed_price = winner_price
                    task.status = "ACCEPTED"
                    self.accepted_contracts[task_id] = task

                    ts = self.env.elapsed_ms() / 1000
                    logging.info(
                        f"[{ts:.3f}s][NEG][{self.color.upper()}] "
                        f"accept oferta task={task_id}, agent={winner_id}, "
                        f"pret={winner_price}, round={task.round_no}"
                    )

                    await self._send_accept(winner_id, task)
                    finished.append(task_id)
                    continue

            if task.round_no < task.max_rounds:
                task.round_no += 1
                task.asking_price += 1
                task.proposals.clear()
                self.cfp_pending[task_id] = now + 0.2

                ts = self.env.elapsed_ms() / 1000
                logging.info(
                    f"[{ts:.3f}s][NEG][{self.color.upper()}] "
                    f"task={task_id} fara oferta buna -> runda {task.round_no}/{task.max_rounds}, "
                    f"asking_price={task.asking_price}"
                )

                await self._broadcast_cfp(
                    task,
                    self._estimate_reward(task.hole_pos, self.env.world),
                    task.asking_price
                )
            else:
                ts = self.env.elapsed_ms() / 1000
                logging.info(
                    f"[{ts:.3f}s][NEG][{self.color.upper()}] "
                    f"task={task_id} esuat dupa 3 runde, revin la plan local"
                )
                task.status = "FAILED"
                self.forced_local_task = task
                finished.append(task_id)
                self.cfp_pending.pop(task_id, None)
                self.open_contracts.pop(task_id, None)
                continue

        for task_id in finished:
            self.cfp_pending.pop(task_id, None)
            task = self.open_contracts.get(task_id)
            if task is not None and task.status == "FAILED":
                self.open_contracts.pop(task_id, None)

    async def _execute_current_contract(self, world: WorldState, agent_state):
        task = self.current_contract
        if task is None:
            return

        ts = self.env.elapsed_ms() / 1000
        logging.info(
            f"[{ts:.3f}s][CONTRACT][{self.color.upper()}] "
            f"execut task={task.task_id}"
        )

        if agent_state.carried_tile is None:
            await self._go_pick_tile(world, agent_state, task.tile_pos, task.tile_color)
            return

        if agent_state.carried_tile != task.tile_color:
            await self.drop_tile()
            return

        nav = find_path_adjacent_to(agent_state.position, task.hole_pos, world, set())
        if nav is None:
            logging.info(
                f"[{self.env.elapsed_ms()/1000:.3f}s][CONTRACT][{self.color.upper()}] "
                f"nu pot ajunge la groapa pentru task={task.task_id}"
            )
            return

        move_path, use_direction = nav
        success = await self._follow_path(move_path)
        if not success:
            return

        result = await self.use_tile(use_direction)
        ts = self.env.elapsed_ms() / 1000
        if result.success:
            logging.info(
                f"[{ts:.3f}s][CONTRACT][{self.color.upper()}] "
                f"task={task.task_id} executat cu succes"
            )
            await self._send_task_done(task.beneficiary_id, task.task_id)
            self.current_contract = None
        else:
            logging.info(
                f"[{ts:.3f}s][CONTRACT][{self.color.upper()}] "
                f"task={task.task_id} a esuat la USE_TILE"
            )

    # Calculează un scor pentru perechea (dală, groapă) pe baza distanței totale
    # și a recompensei obținute; returnează -inf dacă traseul nu există sau groapa e plină.
    def _score_task(
        self,
        tile_pos: Position,
        hole_pos: Position,
        world: WorldState,
    ) -> float:

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

    # Găsește cea mai bună pereche (dală proprie, groapă proprie) prin scorarea tuturor
    # combinațiilor posibile; sare peste dalele mai apropiate de un alt agent liber.
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
        best_prefer_delegate = False

        for tp in own_tiles:
            for hp in own_holes:
                score = self._score_task(tp, hp, world)
                if score == float("-inf"):
                    continue

                my_path = find_path(self._current_pos(world), tp, world, set())
                my_dist = len(my_path) if my_path is not None else 999

                prefer_delegate = False
                closer_agent_id = None

                for other_agent in other_free_agents:
                    their_path = find_path(other_agent.position, tp, world, set())
                    their_dist = len(their_path) if their_path is not None else 999

                    if their_dist + 1 < my_dist:
                        prefer_delegate = True
                        closer_agent_id = other_agent.agent_id
                        break

                if prefer_delegate:
                    ts = self.env.elapsed_ms() / 1000
                    logging.info(
                        f"[{ts:.3f}s][DELEGATE?][{self.color.upper()}] "
                        f"task dala ({tp.x},{tp.y}) -> groapa ({hp.x},{hp.y}); "
                        f"alt agent liber ({closer_agent_id}) pare mai aproape"
                    )

                if score > best_score:
                    best_score = score
                    best_tile = tp
                    best_hole = hp
                    best_prefer_delegate = prefer_delegate

        if best_tile is None:
            return None

        return (best_tile, best_hole, best_score, best_prefer_delegate)

    def _my_state(self):
        return self.env.world.get_agent(self.agent_id)

    def _other_agent_positions(self) -> set:
        return {
            a.position for a in self.env.world.agents
            if a.agent_id != self.agent_id
        }

    # executa lista de miscari si daca esueaza returneaza false pentru
    # replanificare pentru a semnala necesitatea replanificării
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

            # Procesează toate mesajele acumulate în inbox înainte de a lua o decizie
            await self._drain_messages()
            await self._finalize_contracts()

            ts = self.env.elapsed_ms() / 1000
            carried = agent_state.carried_tile if agent_state.carried_tile else "nimic"
            logging.info(
                f"[{ts:.3f}s][TICK][{self.color.upper()}] "
                f"pos=({agent_state.position.x},{agent_state.position.y}) | "
                f"transporta={carried} | "
                f"puncte={agent_state.points}"
            )

            # Obține referința la starea lumii
            world = self.env.world

            if self.current_contract is not None:
                await self._execute_current_contract(world, agent_state)
                await asyncio.sleep(0.01)
                continue
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
    # Faza de colectare: caută cel mai bun task cu dale proprii; dacă nu există,
    # încearcă să ridice orice dală disponibilă pentru a debloca situația.
    async def _phase_collect_tile(self, world: WorldState, agent_state):
        await self._drain_messages()

        ts = self.env.elapsed_ms() / 1000
        logging.info(
            f"[{ts:.3f}s][PLAN][{self.color.upper()}] "
            f"caut cel mai bun task (nu transport nimic)..."
        )

        if self.forced_local_task is not None:
            task = self.forced_local_task
            ts = self.env.elapsed_ms() / 1000
            logging.info(
                f"[{ts:.3f}s][LOCAL][{self.color.upper()}] "
                f"negocierea a esuat pentru task={task.task_id}, il execut singur"
            )
            await self._announce(
                "PICK",
                f"{task.tile_color} tile at ({task.tile_pos.x},{task.tile_pos.y}) "
                f"for hole ({task.hole_pos.x},{task.hole_pos.y}) [forced-local]"
            )
            await self._go_pick_tile(world, agent_state, task.tile_pos, task.tile_color)
            self.forced_local_task = None
            return
        
        if self.cfp_pending:
            logging.info(
                f"[{ts:.3f}s][NEG][{self.color.upper()}] "
                f"am deja o negociere activa, astept raspunsul"
            )
            await asyncio.sleep(0.1)
            return
        

        best = self._best_task(world)

        if best is not None:
            tile_pos, hole_pos, score, prefer_delegate = best
            my_steps = self._estimate_steps(self._current_pos(world), tile_pos, hole_pos, world)
            
            if not prefer_delegate and (score >= 8 or my_steps <= 6):
                ts = self.env.elapsed_ms() / 1000
                logging.info(
                    f"[{ts:.3f}s][TASK][{self.color.upper()}] "
                    f"fac singur taskul: dala {self.color} la ({tile_pos.x},{tile_pos.y}) "
                    f"-> groapa la ({hole_pos.x},{hole_pos.y}) | scor={score:.1f}"
                )
                await self._announce(
                    "PICK",
                    f"{self.color} tile at ({tile_pos.x},{tile_pos.y}) "
                    f"for hole ({hole_pos.x},{hole_pos.y}) [score={score:.1f}]"
                )
                await self._go_pick_tile(world, agent_state, tile_pos, self.color)
                return
            
            if prefer_delegate or score < 8:
                ts = self.env.elapsed_ms() / 1000
                logging.info(
                    f"[{ts:.3f}s][NEG][{self.color.upper()}] "
                    f"alt agent pare mai eficient, pornesc CFP pentru task local"
                )
                existing = None
                for t in self.open_contracts.values():
                    if (
                        t.tile_pos == tile_pos and
                        t.hole_pos == hole_pos and
                        t.status in ("OPEN", "CFP_SENT")
                    ):
                        existing = t
                        break

                if existing is None:
                    task = ContractTask(
                        task_id=self._make_task_id(),
                        beneficiary_id=self.agent_id,
                        beneficiary_color=self.color,
                        executor_id=None,
                        tile_pos=tile_pos,
                        tile_color=self.color,
                        hole_pos=hole_pos,
                        hole_color=self.color,
                        round_no=1,
                        max_rounds=3,
                        asking_price=3,
                        status="CFP_SENT",
                    )

                    self.open_contracts[task.task_id] = task
                    self.cfp_pending[task.task_id] = self.env.elapsed_ms() / 1000 + 0.2

                    ts = self.env.elapsed_ms() / 1000
                    logging.info(
                        f"[{ts:.3f}s][NEG][{self.color.upper()}] "
                        f"task={task.task_id} pornesc runda 1/3, asking_price={task.asking_price}"
                    )

                    await self._broadcast_cfp(
                        task,
                        self._estimate_reward(hole_pos, world),
                        task.asking_price
                    )
                return

        ts = self.env.elapsed_ms() / 1000
        logging.info(
            f"[{ts:.3f}s][IDLE][{self.color.upper()}] "
            f"nu am gasit nici task, nici fallback"
        )
        await asyncio.sleep(0.05)


    # Navighează agentul până la dală, o ridică și anunță ceilalți agenți;
    # dacă dala nu mai e disponibilă la sosire, decrementează rezervarea și replanifică.
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
        # Dacă agentul nu este deja pe poziția dalei, calculează un drum până acolo
        path = find_path(agent_state.position, tile_pos, world, set())
        if path is None:
            ts = self.env.elapsed_ms() / 1000
            logging.info(
                f"[{ts:.3f}s][NO PATH][{self.color.upper()}] "
                f"nu exista drum la ({tile_pos.x},{tile_pos.y}) — astept"
            )
            await asyncio.sleep(0.1)
            return
        # Urmează drumul calculat pas cu pas
        success = await self._follow_path(path)
        if success:
            # ridica dala
            result = await self.pick(color)
            ts = self.env.elapsed_ms() / 1000
            if result.success:
                logging.info(
                    f"[{ts:.3f}s][PICK OK][{self.color.upper()}] "
                    f"am ridicat dala {color} de la ({tile_pos.x},{tile_pos.y})"
                )
                self.reserved_tiles[tile_pos] = max(0, self.reserved_tiles[tile_pos] - 1)
                # Anunță ceilalți agenți că dala a fost ridicată
                await self._announce("PICK_DONE", f"{color} tile at ({tile_pos.x},{tile_pos.y})")
            else:
                logging.info(
                    f"[{ts:.3f}s][PICK FAIL][{self.color.upper()}] "
                    f"dala {color} la ({tile_pos.x},{tile_pos.y}) luata de altcineva — replanific"
                )
                self.reserved_tiles[tile_pos] = max(0, self.reserved_tiles[tile_pos] - 1)

    # Faza de livrare: caută cea mai bună groapă pentru dala transportată, navighează
    # până în celula adiacentă și folosește dala; dacă groapa e inaccesibilă, caută alternativă.
    async def _phase_deliver_tile(self, world: WorldState, agent_state):
        tile_color = agent_state.carried_tile
        ts = self.env.elapsed_ms() / 1000
        logging.info(
            f"[{ts:.3f}s][DELIVER][{self.color.upper()}] "
            f"transport dala {tile_color} — caut cea mai buna groapa"
        )
        # Caută cea mai bună groapă pentru culoarea dalei transportate
        hole_pos = best_hole_for_color(tile_color, world)

        if hole_pos is None:
            ts = self.env.elapsed_ms() / 1000
            logging.info(
                f"[{ts:.3f}s][DROP][{self.color.upper()}] "
                f"nu exista groapa de culoarea {tile_color} — las dala jos"
            )
            await self._announce("DROP", f"{tile_color} tile (no matching hole)")
            await self.drop_tile()
            return

        ts = self.env.elapsed_ms() / 1000
        logging.info(
            f"[{ts:.3f}s][TARGET][{self.color.upper()}] "
            f"groapa tinta gasita la ({hole_pos.x},{hole_pos.y}) pentru dala {tile_color}"
        )
        # Anunță ceilalți agenți
        await self._announce(
            "USE_TILE",
            f"{tile_color} tile -> hole at ({hole_pos.x},{hole_pos.y})"
        )

        # blocked = self._other_agent_positions()
        # Caută un drum până într-o celulă vecină gropii
        nav = find_path_adjacent_to(agent_state.position, hole_pos, world, set())
        # Dacă groapa aleasă nu este accesibilă, încearcă altă groapă de aceeași culoare
        if nav is None:
            ts = self.env.elapsed_ms() / 1000
            logging.info(
                f"[{ts:.3f}s][REROUTE][{self.color.upper()}] "
                f"groapa ({hole_pos.x},{hole_pos.y}) inaccesibila — caut alternativa"
            )
            for hole in world.get_active_holes():
                if hole.position == hole_pos or hole.color != tile_color:
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
        # Dacă nici după rerutare nu există o groapă accesibilă, lasă dala jos
        if nav is None:
            ts = self.env.elapsed_ms() / 1000
            logging.info(
                f"[{ts:.3f}s][DROP][{self.color.upper()}] "
                f"nicio groapa accesibila — las dala jos și replanific"
            )
            await self.drop_tile()
            return

        move_path, use_direction = nav
        # Urmează drumul spre groapă
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

    # Golește complet inbox-ul, procesând toate mesajele acumulate; actualizează
    # contorul de rezervări pentru PICK/UNBLOCK_PICK și îl decrementează la PICK_DONE.
    async def _drain_messages(self):
        # Procesează toate mesajele din inbox, nu doar unul.
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
            # Dacă alt agent a anunțat că vrea să ridice o dală,
            # atunci această dală este marcată ca rezervată
            if action in ("PICK", "UNBLOCK_PICK"):
                match = re.search(r'\((\d+),(\d+)\)', details)
                if match:
                    x, y = int(match.group(1)), int(match.group(2))
                    self.reserved_tiles[Position(x, y)] += 1
                    logging.info(
                        f"[{ts:.3f}s][RESERVE][{self.color.upper()}] "
                        f"dala la ({x},{y}) rezervata de {sender_color.upper()}"
                    )
            # Dacă alt agent anunță că a ridicat deja dala,
            # eliberăm rezervarea de pe acea poziție
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
            elif action == "CFP":
                task_id = msg["content"]["task_id"]
                beneficiary_id = msg["content"]["beneficiary_id"]
                tile_x, tile_y = msg["content"]["tile_pos"]
                hole_x, hole_y = msg["content"]["hole_pos"]
                tile_color = msg["content"]["tile_color"]
                hole_color = msg["content"]["hole_color"]
                expected_reward = msg["content"]["expected_reward"]
                asking_price = msg["content"]["asking_price"]
                round_no = msg["content"]["round_no"]

                if beneficiary_id != self.agent_id:
                    my_steps = self._estimate_steps(
                        self._current_pos(self.env.world),
                        Position(tile_x, tile_y),
                        Position(hole_x, hole_y),
                        self.env.world,
                    )

                    if my_steps < 999:
                        my_cost = max(1, math.ceil(my_steps * 0.5))

                        if asking_price >= my_cost and asking_price <= expected_reward - 1:
                            ts = self.env.elapsed_ms() / 1000
                            logging.info(
                                f"[{ts:.3f}s][NEG][{self.color.upper()}] "
                                f"accept sa licitez task={task_id}, round={round_no}, "
                                f"ask={asking_price}, cost={my_cost}"
                            )
                            await self._send_proposal(beneficiary_id, task_id, asking_price)
            elif action == "PROPOSE":
                task_id = msg["content"]["task_id"]
                price = msg["content"]["price"]
                sender_id = msg["from"]

                task = self.open_contracts.get(task_id)
                if task is not None:
                    task.proposals[sender_id] = price
                    logging.info(
                        f"[{ts:.3f}s][NEG][{self.color.upper()}] "
                        f"am primit oferta de la agent={sender_id} pentru task={task_id}: {price}"
                    )
            elif action == "ACCEPT":
                task_id = msg["content"]["task_id"]
                beneficiary_id = msg["content"]["beneficiary_id"]
                tile_x, tile_y = msg["content"]["tile_pos"]
                hole_x, hole_y = msg["content"]["hole_pos"]
                tile_color = msg["content"]["tile_color"]
                hole_color = msg["content"]["hole_color"]
                price = msg["content"]["price"]

                task = ContractTask(
                    task_id=task_id,
                    beneficiary_id=beneficiary_id,
                    beneficiary_color=hole_color,
                    executor_id=self.agent_id,
                    tile_pos=Position(tile_x, tile_y),
                    tile_color=tile_color,
                    hole_pos=Position(hole_x, hole_y),
                    hole_color=hole_color,
                    agreed_price=price,
                    status="EXECUTING",
                )

                self.current_contract = task
                self.contract_wait_until = 0

                logging.info(
                    f"[{ts:.3f}s][NEG][{self.color.upper()}] "
                    f"am primit ACCEPT pentru task={task_id}, pret={price}"
                )

            elif action == "TASK_DONE":
                task_id = msg["content"]["task_id"]

                task = self.accepted_contracts.get(task_id)
                if task is not None and task.beneficiary_id == self.agent_id:
                    logging.info(
                        f"[{ts:.3f}s][NEG][{self.color.upper()}] "
                        f"task={task_id} finalizat, transfer {task.agreed_price} puncte"
                    )

                    await self.transfer_points(task.executor_id, task.agreed_price)
                    task.status = "DONE"

                    self.accepted_contracts.pop(task_id, None)
                    self.open_contracts.pop(task_id, None)
                    self.current_contract = None