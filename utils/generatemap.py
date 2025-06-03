import random
from collections import deque

def line_points(x0, y0, x1, y1):
    """Bresenham's line algorithm: return list of points between (x0,y0) and (x1,y1)"""
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    if dx > dy:
        err = dx // 2
        while x != x1:
            points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy // 2
        while y != y1:
            points.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    points.append((x1, y1))
    return points

def generate_roads(width, height, road_count=30):
    """Generate a grid with roads (0) and empty land (1) with random irregular roads"""
    grid = [[1 for _ in range(width)] for _ in range(height)]

    # Add base grid roads every 6 tiles to create dense city blocks
    grid_spacing = 6
    for x in range(0, width, grid_spacing):
        for y in range(height):
            grid[y][x] = 0  # vertical grid roads
    for y in range(0, height, grid_spacing):
        for x in range(width):
            grid[y][x] = 0  # horizontal grid roads

    for _ in range(road_count):
        # Pick random start point on border
        edge = random.choice(['top', 'bottom', 'left', 'right'])
        if edge in ('top', 'bottom'):
            x0 = random.randint(0, width - 1)
            y0 = 0 if edge == 'top' else height - 1
        else:
            y0 = random.randint(0, height - 1)
            x0 = 0 if edge == 'left' else width - 1

        # Pick random end point on another border
        other_edges = [e for e in ['top', 'bottom', 'left', 'right'] if e != edge]
        edge2 = random.choice(other_edges)
        if edge2 in ('top', 'bottom'):
            x1 = random.randint(0, width - 1)
            y1 = 0 if edge2 == 'top' else height - 1
        else:
            y1 = random.randint(0, height - 1)
            x1 = 0 if edge2 == 'left' else width - 1

        # More segments for jagged roads to create more intersections
        num_segments = random.randint(3, 6)
        points = [(x0, y0)]
        for _ in range(num_segments - 1):
            px = random.randint(0, width - 1)
            py = random.randint(0, height - 1)
            points.append((px, py))
        points.append((x1, y1))

        # Draw road lines between points with random width (1-2 tiles)
        for i in range(len(points) - 1):
            p0 = points[i]
            p1 = points[i + 1]
            line = line_points(*p0, *p1)
            road_width = random.randint(1, 2)
            for (lx, ly) in line:
                for wx in range(-road_width // 2, road_width // 2 + 1):
                    for wy in range(-road_width // 2, road_width // 2 + 1):
                        rx, ry = lx + wx, ly + wy
                        if 0 <= rx < width and 0 <= ry < height:
                            grid[ry][rx] = 0  # Road tile

    return grid

def flood_fill(grid, x, y, visited):
    """Flood fill to find connected empty plot (building footprint)"""
    width = len(grid[0])
    height = len(grid)
    q = deque()
    q.append((x, y))
    plot = []
    while q:
        cx, cy = q.popleft()
        if (cx, cy) in visited:
            continue
        if not (0 <= cx < width and 0 <= cy < height):
            continue
        if grid[cy][cx] != 1:
            continue
        visited.add((cx, cy))
        plot.append((cx, cy))
        for nx, ny in [(cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)]:
            q.append((nx, ny))
    return plot

def find_building_plots(grid):
    """Find all connected land plots (potential building footprints)"""
    visited = set()
    plots = []
    height = len(grid)
    width = len(grid[0])
    for y in range(height):
        for x in range(width):
            if grid[y][x] == 1 and (x, y) not in visited:
                plot = flood_fill(grid, x, y, visited)
                if len(plot) >= 4:  # ignore tiny plots
                    plots.append(plot)
    return plots

def point_neighbors_4(p):
    x, y = p
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

def generate_building_layers(width, height, plots, max_height=5, min_height=1):
    """
    Generate a 3D layers grid for buildings from plots.
    Buildings fully fill their plot tiles.
    Each building plot is assigned a style:
    - Original: brown bricks (1) and brown windows (2), with sparse propaganda tiles (5-8)
    - Gray: gray bricks (3) and gray windows (4), no propaganda
    - Tile9: all tiles are 9, no propaganda
    """
    layers = [[[0 for _ in range(width)] for _ in range(height)] for _ in range(max_height)]

    propaganda_tiles = [5, 6, 7, 8]
    PROPAGANDA_CHANCE = 0.08  # 8% chance for propaganda on edge tiles (brown style only)

    for plot in plots:
        # Decide building height randomly per plot, respecting min_height
        building_height = random.randint(min_height, max_height)
        plot_set = set(plot)

        # Decide style randomly: True = brown/original, False = gray, "tile9" = new type without propaganda
        original_style = random.choice([True, False, "tile9"])

        def is_edge_tile(x, y):
            for nx, ny in point_neighbors_4((x, y)):
                if nx < 0 or nx >= width or ny < 0 or ny >= height:
                    return True
                if (nx, ny) not in plot_set:
                    return True
            return False

        for (x, y) in plot:
            for layer in range(building_height):
                if original_style == "tile9":
                    # Third style: all tile 9, no propaganda
                    tile = 9
                else:
                    if is_edge_tile(x, y):
                        if original_style:
                            # Brown buildings get sparse propaganda
                            if random.random() < PROPAGANDA_CHANCE:
                                tile = random.choice(propaganda_tiles)
                            else:
                                tile = 2 if random.random() < 0.3 else 1
                        else:
                            # Gray buildings no propaganda
                            tile = 4 if random.random() < 0.3 else 3
                    else:
                        # Interior solid bricks
                        tile = 1 if original_style else 3

                layers[layer][y][x] = tile

    return layers

def save_map_to_file(layers, filename="city_map.ussa3d"):
    with open(filename, "w") as f:
        for layer_idx, layer in enumerate(layers):
            for row in layer:
                f.write(",".join(str(tile) for tile in row) + "\n")
            if layer_idx != len(layers) - 1:
                f.write("\n")
    print(f"Map saved to {filename}")

if __name__ == "__main__":
    WIDTH = 48*10
    HEIGHT = 48*10
    MAX_LAYERS = 22
    MIN_LAYERS = 7
    ROAD_COUNT = 10

    print("Generating roads...")
    grid = generate_roads(WIDTH, HEIGHT, ROAD_COUNT)

    print("Finding building plots...")
    plots = find_building_plots(grid)
    print(f"Found {len(plots)} building plots")

    print("Generating building layers...")
    building_layers = generate_building_layers(WIDTH, HEIGHT, plots, MAX_LAYERS, MIN_LAYERS)

    print("Saving map...")
    save_map_to_file(building_layers, "city_map.ussa3d")
