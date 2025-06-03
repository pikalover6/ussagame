import pygame
import math
import sys
import numpy as np

# === FULLSCREEN & RENDER SCALE SETTINGS ===
FULLSCREEN = True
RENDER_SCALE = 0.5  # fraction of fullscreen resolution to render at internally, e.g. 0.5 = half resolution

# === FIELD‑OF‑VIEW SETTINGS ===
FOV = math.pi / 1.1           # ~163° – change freely at runtime, just be sure to call update_ray_params()
HALF_FOV = FOV / 2

# === WORLD / PLAYER CONSTANTS ===
TILE = 50
CAMERA_HEIGHT = 0-TILE          # <- 1 full wall‑unit above the floor
MAX_WALL_HEIGHT = 10
GRAVITY = 0.8
JUMP_SPEED = 12
MAX_DEPTH = 20
INFINITE_WORLD = False
FLOOR_TILE = TILE * 16

# === RAYCAST PARAMS – filled in once the final render width is known ===
NUM_RAYS = None
DELTA_ANGLE = None
DIST_PLANE = None
SCALE = 1
COS_REL = None
SIN_REL = None

# ------------------------------------------------------------------
#   INITIALISATION HELPERS
# ------------------------------------------------------------------

def update_ray_params(screen_width: int):
    """Re‑calculate all derived arrays when the render width OR the FOV changes."""
    global NUM_RAYS, DELTA_ANGLE, DIST_PLANE, COS_REL, SIN_REL

    NUM_RAYS = screen_width // SCALE
    DELTA_ANGLE = FOV / NUM_RAYS
    DIST_PLANE = (screen_width / 2) / math.tan(HALF_FOV)

    rel_angles = np.linspace(-HALF_FOV, HALF_FOV, NUM_RAYS, endpoint=True)
    COS_REL = np.cos(rel_angles)
    SIN_REL = np.sin(rel_angles)


def procedural_height(mx, my):
    return (hash((mx, my)) & 0xFFFFFFFF) % (MAX_WALL_HEIGHT + 1)


def get_tile(mx, my):
    if INFINITE_WORLD:
        return procedural_height(mx, my)
    static_map = [
        [1, 2, 3, 0, 0, 3, 2, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [4, 0, 0, 0, 9, 0, 0, 3],
        [1, 0, 4, 4, 9, 10, 0, 1],
        [1, 1, 1, 0, 0, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [3, 0, 0, 0, 0, 0, 0, 4],
        [1, 2, 3, 0, 0, 3, 2, 1],
    ]
    if 0 <= mx < 8 and 0 <= my < 8:
        return static_map[my][mx]
    return 0

# ------------------------------------------------------------------
#   PLAYER
# ------------------------------------------------------------------

class Player:
    def __init__(self):
        self.x = TILE * 1.5
        self.y = TILE * 1.5
        self.angle = 0.0
        self.speed = 2.5
        self.height = 0.0          # dynamic jump offset
        self.vert_velocity = 0.0
        self.pitch = 0.0           # looking up / down
        self.cam_height = CAMERA_HEIGHT

    # --------------------------------------------------------------
    #   Movement & physics
    # --------------------------------------------------------------
    def movement(self):
        keys = pygame.key.get_pressed()
        sin_a, cos_a = math.sin(self.angle), math.cos(self.angle)

        if keys[pygame.K_w]:
            nx, ny = self.x + cos_a * self.speed, self.y + sin_a * self.speed
            if get_tile(int(nx // TILE), int(ny // TILE)) == 0:
                self.x, self.y = nx, ny
        if keys[pygame.K_s]:
            nx, ny = self.x - cos_a * self.speed, self.y - sin_a * self.speed
            if get_tile(int(nx // TILE), int(ny // TILE)) == 0:
                self.x, self.y = nx, ny
        if keys[pygame.K_a]:
            self.angle -= 0.04
        if keys[pygame.K_d]:
            self.angle += 0.04
        self.angle %= math.tau

        # jumping ---------------------------------------------------
        if keys[pygame.K_SPACE] and self.height == 0:
            self.vert_velocity = JUMP_SPEED
        self.vert_velocity -= GRAVITY
        self.height = max(0.0, self.height + self.vert_velocity)
        if self.height == 0:
            self.vert_velocity = 0.0

# ------------------------------------------------------------------
#   TEXTURES & FLOOR BUFFER
# ------------------------------------------------------------------

def load_textures():
    textures = {}
    column_cache = {}

    for h in range(1, MAX_WALL_HEIGHT + 1):
        try:
            img = pygame.image.load(f"texture_{h}.jpg").convert()
        except Exception:
            colour = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)][(h - 1) % 4]
            img = pygame.Surface((TILE, TILE))
            img.fill(colour)
        img = pygame.transform.scale(img, (TILE, TILE))
        textures[h] = img
        column_cache[h] = [img.subsurface(x, 0, 1, TILE) for x in range(TILE)]

    try:
        floor_img = pygame.image.load("texture_1.jpg").convert()
    except Exception:
        floor_img = pygame.Surface((FLOOR_TILE, FLOOR_TILE))
        floor_img.fill((100, 100, 100))

    floor_tex = pygame.transform.scale(floor_img, (FLOOR_TILE, FLOOR_TILE))
    floor_arr = pygame.surfarray.array3d(floor_tex)
    return textures, column_cache, floor_arr

# ------------------------------------------------------------------
#   RAYCASTER
# ------------------------------------------------------------------

def ray_cast(screen: pygame.Surface, player: Player, col_cache: dict):
    px, py = player.x, player.y

    cos_p, sin_p = math.cos(player.angle), math.sin(player.angle)

    for ray in range(NUM_RAYS):
        # 1) Build the direction of this ray -----------------------
        dir_x = cos_p * COS_REL[ray] - sin_p * SIN_REL[ray]
        dir_y = sin_p * COS_REL[ray] + cos_p * SIN_REL[ray]

        # 2) Digital‑differential analyser (grid walk) -------------
        map_x, map_y = int(px // TILE), int(py // TILE)
        delta_x = abs(TILE / dir_x) if dir_x != 0 else float("inf")
        delta_y = abs(TILE / dir_y) if dir_y != 0 else float("inf")

        if dir_x < 0:
            step_x = -1
            side_x = (px - map_x * TILE) * delta_x / TILE
        else:
            step_x = 1
            side_x = ((map_x + 1) * TILE - px) * delta_x / TILE

        if dir_y < 0:
            step_y = -1
            side_y = (py - map_y * TILE) * delta_y / TILE
        else:
            step_y = 1
            side_y = ((map_y + 1) * TILE - py) * delta_y / TILE

        hit_height = 0
        side = 0

        while hit_height == 0 and 0 <= map_x < 256 and 0 <= map_y < 256:
            if side_x < side_y:
                side_x += delta_x
                map_x += step_x
                side = 0
            else:
                side_y += delta_y
                map_y += step_y
                side = 1
            hit_height = get_tile(map_x, map_y)
            if hit_height or abs(map_x - px / TILE) > MAX_DEPTH or abs(map_y - py / TILE) > MAX_DEPTH:
                break
        if hit_height == 0:
            continue

        # 3) Exact wall distance -----------------------------------
        if side == 0:
            ray_len = (map_x - px / TILE + (1 - step_x) / 2) * TILE / dir_x
            wall_x = (py / TILE) + ray_len * dir_y / TILE
        else:
            ray_len = (map_y - py / TILE + (1 - step_y) / 2) * TILE / dir_y
            wall_x = (px / TILE) + ray_len * dir_x / TILE
        wall_x -= math.floor(wall_x)
        tex_x = int(wall_x * TILE)

        # Convert to perpendicular distance using dot‑product with camera dir
        cos_rel = dir_x * cos_p + dir_y * sin_p  # == cos(angle between ray & view)
        perp = max(0.0001, ray_len * cos_rel)

        # 4) Projected wall slice size -----------------------------
        wall_world_height = hit_height * TILE
        line_h = int((wall_world_height * DIST_PLANE) / perp)

        # 5) Vertical placement (camera elevated!) -----------------
        pitch_offset = math.tan(player.pitch) * DIST_PLANE
        cam_offset = int(player.cam_height * DIST_PLANE / perp)

        floor_line = SCREEN_HEIGHT + int(pitch_offset) - cam_offset
        y_pos = floor_line - line_h  # top pixel of the slice

        # 6) Draw column -------------------------------------------
        column = col_cache[hit_height][tex_x]
        column = pygame.transform.scale(column, (SCALE, line_h))
        screen.blit(column, (ray * SCALE, y_pos))

# ------------------------------------------------------------------
#   FLOOR / CEILING SHADER
# ------------------------------------------------------------------

def draw_floor(screen: pygame.Surface, player: Player, floor_arr: np.ndarray):
    buf = pygame.surfarray.pixels3d(screen)

    # ---- SKY -----------------------------------------------------
    buf[:, :SCREEN_HEIGHT // 2] = (100, 149, 237)

    # ---- Pre‑compute edge ray dir (needed for floor‑casting) -----
    ang0 = player.angle - HALF_FOV
    ang1 = player.angle + HALF_FOV
    dx0, dy0 = math.cos(ang0), math.sin(ang0)
    dx1, dy1 = math.cos(ang1), math.sin(ang1)

    xs = np.arange(SCREEN_WIDTH)

    # camera height (now includes elevated eye‑level!)
    cam_z = TILE / 2 + player.height + player.cam_height

    pitch_offset = math.tan(player.pitch) * DIST_PLANE

    for y in range(SCREEN_HEIGHT // 2, SCREEN_HEIGHT):
        p = y - (SCREEN_HEIGHT / 2 + pitch_offset)
        if p == 0:
            p = 0.0001

        row_distance = cam_z * DIST_PLANE / p

        fx = player.x + row_distance * dx0 + (row_distance * (dx1 - dx0) / SCREEN_WIDTH) * xs
        fy = player.y + row_distance * dy0 + (row_distance * (dy1 - dy0) / SCREEN_WIDTH) * xs

        tx = (fx.astype(np.int32) % FLOOR_TILE)
        ty = (fy.astype(np.int32) % FLOOR_TILE)
        buf[:, y] = floor_arr[tx, ty]

    del buf

# ------------------------------------------------------------------
#   MAIN LOOP
# ------------------------------------------------------------------

def main():
    pygame.init()
    pygame.mouse.set_visible(False)
    pygame.event.set_grab(True)

    info = pygame.display.Info()
    native_w, native_h = info.current_w, info.current_h

    flags = pygame.FULLSCREEN | pygame.SCALED if FULLSCREEN else 0
    display_surface = pygame.display.set_mode((native_w, native_h), flags)

    # --------------------------------------------------------------
    #   Render resolution (internal) -------------------------------
    global SCREEN_WIDTH, SCREEN_HEIGHT
    SCREEN_WIDTH = int(native_w * RENDER_SCALE)
    SCREEN_HEIGHT = int(native_h * RENDER_SCALE)

    update_ray_params(SCREEN_WIDTH)  # derive NUM_RAYS etc.

    game_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))

    player = Player()

    _, col_cache, floor_arr = load_textures()

    window_w, window_h = display_surface.get_size()
    scale = min(window_w // SCREEN_WIDTH, window_h // SCREEN_HEIGHT)
    scaled_w = SCREEN_WIDTH * scale
    scaled_h = SCREEN_HEIGHT * scale
    offset_x = (window_w - scaled_w) // 2
    offset_y = (window_h - scaled_h) // 2

    clock = pygame.time.Clock()

    MOUSE_SENSITIVITY_X = 0.002
    MOUSE_SENSITIVITY_Y = 0.002

    while True:
        # ----------------------------------------------------------
        #   Event handling
        # ----------------------------------------------------------
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()

        # ----------------------------------------------------------
        #   Mouse look
        # ----------------------------------------------------------
        window_center = (window_w // 2, window_h // 2)
        pygame.mouse.set_pos(window_center)

        mouse_dx, mouse_dy = pygame.mouse.get_rel()
        player.angle += mouse_dx * MOUSE_SENSITIVITY_X
        player.angle %= math.tau

        player.pitch -= mouse_dy * MOUSE_SENSITIVITY_Y
        player.pitch = max(-math.pi / 2, min(math.pi / 2, player.pitch))

        # ----------------------------------------------------------
        #   Movement / physics
        # ----------------------------------------------------------
        player.movement()

        # ----------------------------------------------------------
        #   Rendering  🎨
        # ----------------------------------------------------------
        game_surface.fill(0)
        draw_floor(game_surface, player, floor_arr)
        ray_cast(game_surface, player, col_cache)

        # upscale to the window's back‑buffer
        upscaled = pygame.transform.scale(game_surface, (scaled_w, scaled_h))
        display_surface.fill((0, 0, 0))
        display_surface.blit(upscaled, (offset_x, offset_y))

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()
