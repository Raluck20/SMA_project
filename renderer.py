from world_state import WorldState
from models import Position

DISPLAY_INTERVAL_MS = 250
CELL_WIDTH = 4

def _color_abbrev(color: str, length: int = 3) -> str:
    return color[:length].capitalize()

def render_grid(world: WorldState, elapsed_ms: float) -> str:

    lines = []
    elapsed_s = round(elapsed_ms / 1000)
    lines.append(f"Time: {elapsed_s}s")
    lines.append("")

    W = world.width
    H = world.height

    header = "    "
    for x in range(W):
        header += f"{x:<{CELL_WIDTH}} "
    lines.append(header.rstrip())

    def h_sep():
        return "  +" + (" " * CELL_WIDTH + "+") * W

    for y in range(H):
        lines.append(h_sep())

        row1 = f"{y:<2} "
        row2 = "   "

        for x in range(W):
            pos = Position(x, y)
            cell1, cell2 = _render_cell(world, pos)
            row1 += f"{cell1:<{CELL_WIDTH}} "
            row2 += f"{cell2:<{CELL_WIDTH}} "

        lines.append(row1.rstrip())
        lines.append(row2.rstrip())

    lines.append(h_sep())
    lines.append("")

    for agent in world.agents:
        carries = f"carries {agent.carried_tile}" if agent.carried_tile else "carries nothing"
        lines.append(f"{agent.color.capitalize()} agent: {agent.points} points; {carries}")

    return "\n".join(lines)

def _render_cell(world: WorldState, pos: Position) -> tuple[str, str]:

    if pos in world.obstacles:
        return "////", "////"

    active_holes = [h for h in world.holes if h.position == pos and not h.is_filled]
    if active_holes:
        hole = active_holes[0]
        abbrev = _color_abbrev(hole.color, 4)
        return f"#{hole.depth}", abbrev

    agents_here = [a for a in world.agents if a.position == pos]

    tile_stacks = world.get_tiles_at(pos)

    line1_parts = []
    line2_parts = []

    for agent in agents_here:
        abbrev = _color_abbrev(agent.color, 3)
        line1_parts.append(f"@{abbrev}")
        if agent.carried_tile:
            line2_parts.append(_color_abbrev(agent.carried_tile, 3))

    for stack in tile_stacks:
        if stack.count > 0:
            abbrev = _color_abbrev(stack.color, 2)
            line1_parts.append(f"${stack.count}{abbrev}")

    line1 = " ".join(line1_parts) if line1_parts else ""
    line2 = " ".join(line2_parts) if line2_parts else ""

    return line1[:CELL_WIDTH], line2[:CELL_WIDTH]

def render_status_line(world: WorldState, elapsed_ms: float) -> str:
    elapsed_s = elapsed_ms / 1000
    active_holes = len(world.get_active_holes())
    total_pts = sum(a.points for a in world.agents)
    return (
        f"[{elapsed_s:.1f}s] "
        f"Holes remaining: {active_holes} | "
        f"Total points: {total_pts}"
    )