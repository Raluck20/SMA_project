import sys
import asyncio
import logging

from parser import load_system, ParseError
from world_state import build_world
from environment import Environment
from agent import TileWorldAgent, MessageBus
from renderer import render_grid, DISPLAY_INTERVAL_MS


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="[%H:%M:%S]",
)


async def display_loop(env: Environment):
    while env.running:
        await asyncio.sleep(DISPLAY_INTERVAL_MS / 1000)

        # Afișează log-ul de operații + mesaje acumulat înainte de grid
        if env.operation_log:
            print("\n".join(env.operation_log))
            env.operation_log.clear()

        print("=" * 51)
        print(render_grid(env.world, env.elapsed_ms()))
        print("=" * 51)


async def run_simulation(config):
    env = Environment(config)
    world = env.world

    n = len(config.agents)
    bus = MessageBus(n_agents=n)
    all_ids = [a.agent_id for a in config.agents]

    agents = [
        TileWorldAgent(
            agent_id=a.agent_id,
            color=a.color,
            env=env,
            bus=bus,
            all_agent_ids=all_ids,
        )
        for a in config.agents
    ]

    # Afișare inițială
    print("=" * 51)
    print(render_grid(world, 0))
    print("=" * 51)

    tasks = [
        asyncio.create_task(env.run()),
        asyncio.create_task(display_loop(env)),
    ]
    for agent in agents:
        tasks.append(asyncio.create_task(agent.run()))

    await asyncio.gather(*tasks, return_exceptions=True)

    # Afișare finală
    if env.operation_log:
        print("\n".join(env.operation_log))
        env.operation_log.clear()

    print("\n" + "=" * 51)
    print("SIMULATION COMPLETE")
    print(render_grid(env.world, env.elapsed_ms()))
    print("=" * 51)
    print("\nFinal scores:")
    for a in env.world.agents:
        print(f"  {a.color.capitalize()}: {a.points} points")


def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else "system.txt"
    try:
        config = load_system(input_path)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        return 1
    except ParseError as exc:
        print(f"[PARSE ERROR] {exc}")
        return 1

    asyncio.run(run_simulation(config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())