_C=None
_B=True
_A=False

import pygame, math, sys, numpy as np, random, os, time
from ollama import chat, ChatResponse
import numba

FULLSCREEN=_B
RENDER_SCALE=.125
FOV_DEFAULT=math.pi/2
ZOOMED_FOV=math.pi/16
ZOOM_SPEED=3.
TILE=50
CAM_HEIGHT=-TILE//20
GRAVITY, JUMP_SPD = .8*200, 12*8
MAX_DEPTH, FLOOR_TILE = 20, TILE*16
CHUNK_RADIUS_TILES = 256
DIALOGUE_BOX_HEIGHT = 70
NPC_SCALE = 0.725

NUM_RAYS = DELTA_ANGLE = DIST_PLANE = COS_REL = SIN_REL = _C
LOADED_MAP = []
USE_FILE_MAP = _A
CURRENT_LAYER = 0
depth_buffer = _C
show_fps = _A
FONT = _C
dialogue_active = _A
input_active = _A
input_text = ''
ai_response = ''
scroll_offset = 0
disable_propoganda = False
script_dir = os.path.dirname(os.path.abspath(__file__))
textures_folder = os.path.join(script_dir, "textures")
print(textures_folder)

with open('lore.txt', 'r', encoding='utf-8') as file:
    lore = file.read()

with open('intro_lore.txt', 'r', encoding='utf-8') as file:
    intro_lore = file.read()

def update_ray_params(w):
    global NUM_RAYS, DELTA_ANGLE, DIST_PLANE, COS_REL, SIN_REL
    NUM_RAYS = int(w)
    DELTA_ANGLE = FOV_DEFAULT / NUM_RAYS
    DIST_PLANE = w / 2 / math.tan(FOV_DEFAULT / 2)
    a = np.arctan(np.linspace(-1, 1, NUM_RAYS) * math.tan(FOV_DEFAULT / 2))
    COS_REL, SIN_REL = np.cos(a), np.sin(a)

def load_ussa_map(fn):
    global LOADED_MAP, USE_FILE_MAP, CURRENT_LAYER
    try:
        with open(fn) as f:
            layers, layer = [], []
            for line in f:
                line = line.strip()
                if line == '':
                    if layer: layers.append(layer)
                    layer = []
                else:
                    layer.append(list(map(int, line.split(','))))
            if layer: layers.append(layer)
        LOADED_MAP, USE_FILE_MAP, CURRENT_LAYER = layers, _B, 0
        print(f"Loaded map {fn} with {len(LOADED_MAP)} layers.")
    except Exception as e:
        print(f"Failed to load map {fn}: {e}")
        USE_FILE_MAP = _A

def get_tile(mx, my, mz=_C):
    if USE_FILE_MAP:
        z = CURRENT_LAYER if mz is _C else mz
        if 0 <= z < len(LOADED_MAP) and 0 <= my < len(LOADED_MAP[z]) and 0 <= mx < len(LOADED_MAP[z][my]):
            return LOADED_MAP[z][my][mx]
        return 0
    static_map = [[0]*16 for _ in range(16)]
    return static_map[my][mx] if 0 <= mx < 16 and 0 <= my < 16 else 0

class Player:
    def __init__(self):
        A = .0
        self.x, self.y = TILE*1.5, TILE*1.5
        self.angle = A
        self.speed = 150
        self.height = A
        self.vert_velocity = A
        self.pitch = A
        self.cam_height = CAM_HEIGHT
        self.layer = 0
        self.inventory = []
        self.inventory_index = 0  
        self.held_item = None
        self.is_moving = False

    def select_item(self, index, item_textures):
        if 0 <= index < len(self.inventory):
            self.inventory_index = index
            item_key = self.inventory[index]
            self.held_item = item_textures.get(item_key, None)
        else:
            self.held_item = None

    def movement(self, dt):
            keys = pygame.key.get_pressed()
            s, a = math.sin(self.angle), math.cos(self.angle)
            ms = self.speed * dt
            moved = False
            for dx, dy, k in [(a, s, pygame.K_w), (-a, -s, pygame.K_s), (-s, a, pygame.K_d), (s, -a, pygame.K_a)]:
                if keys[k]:
                    nx, ny = self.x + dx * ms, self.y + dy * ms
                    if get_tile(int(nx // TILE), int(ny // TILE), self.layer) == 0:
                        self.x, self.y = nx, ny
                        moved = True
            self.is_moving = moved  # set flag here

            if keys[pygame.K_SPACE] and self.height == 0:
                self.vert_velocity = JUMP_SPD
            self.vert_velocity -= GRAVITY * dt
            self.height = max(0, self.height + self.vert_velocity * dt)
            if self.height == 0:
                self.vert_velocity = 0

# ── 1. load_textures  ──────────────────────────────────────────────────────────
def load_textures():
    tex, col_cache = {}, {}
    colors = [(255,255,255), (139,69,19), (135,206,235), (169,169,169)]
    scale_factor = 0.375  # downscale textures to 50% for speed

    for tid in range(21):
        try:
            fn = f"{textures_folder}/texture_{tid}.jpg"
            if disable_propoganda and tid > 4:
                fn = f"{textures_folder}/texture_1.jpg"
            img = pygame.image.load(fn).convert()
        except:
            try:
                img = pygame.image.load(f"{textures_folder}/texture_{tid}.png").convert()
            except:
                img = pygame.Surface((TILE, TILE))
                img.fill(colors[tid % 4])

        # Pre-scale textures here
        scaled_w = int(TILE * scale_factor)
        scaled_h = int(TILE * scale_factor)
        img = pygame.transform.scale(img, (scaled_w, scaled_h))

        tex[tid] = img

        # Pre-extract vertical 1-pixel slices at scaled resolution for all columns
        col_cache[tid] = [img.subsurface(x, 0, 1, scaled_h).copy() for x in range(scaled_w)]

    # floor texture
    try:
        floor = pygame.image.load(f'{textures_folder}/texture_1.jpg').convert()
    except:
        floor = pygame.Surface((TILE, TILE))
        floor.fill((100, 100, 100))

    floor_scaled = pygame.transform.scale(floor, (int(TILE*scale_factor), int(TILE*scale_factor)))
    floor_arr = pygame.surfarray.array3d(floor_scaled)  # smaller texture array

    return tex, col_cache, floor_arr, scale_factor

def load_npc_textures():
    npc_textures = []
    for i in range(1, 5):
        try:
            img = pygame.image.load(f"{textures_folder}/npc_{i}.png").convert()
            print(f"Loaded NPC texture: {textures_folder}/npc_{i}.png")
        except:
            try:
                img = pygame.image.load(f'{textures_folder}/npc_1.png').convert()
                print(f"Loaded NPC texture: {textures_folder}/npc_1.png (fallback)")
            except:
                img = pygame.Surface((TILE, TILE))
                img.fill((255, 0, 255))
                print("Failed to load NPC texture, using placeholder magenta.")
        img.set_colorkey((0, 0, 0))
        npc_textures.append(pygame.transform.scale(img, (TILE, TILE)))
    return npc_textures

class NPC:
    def __init__(self, x, y, texture):
        self.x, self.y = x, y
        self.angle = random.uniform(0, math.tau)
        self.speed = 60
        self.texture = texture  # single texture surface
        self.destination = self._choose_destination()

    def _choose_destination(self):
        for _ in range(50):
            dx, dy = random.randint(-5, 5), random.randint(-5, 5)
            tx, ty = int(self.x // TILE) + dx, int(self.y // TILE) + dy
            if get_tile(tx, ty, 0) == 0:
                return tx * TILE + TILE / 2, ty * TILE + TILE / 2
        return self.x, self.y

    def update(self, dt):
        tx, ty = self.destination
        dx, dy = tx - self.x, ty - self.y
        dist = math.hypot(dx, dy)
        if dist < 2:
            self.destination = self._choose_destination()
            return
        self.angle = math.atan2(dy, dx)
        nx = self.x + math.cos(self.angle) * self.speed * dt
        ny = self.y + math.sin(self.angle) * self.speed * dt
        if get_tile(int(nx // TILE), int(self.y // TILE), 0) == 0:
            self.x = nx
        if get_tile(int(self.x // TILE), int(ny // TILE), 0) == 0:
            self.y = ny

def ray_cast(screen, player, col_cache, scale_factor):
    global depth_buffer

    px, py = player.x, player.y
    c, a = math.cos(player.angle), math.sin(player.angle)
    ptx, pty = int(px // TILE), int(py // TILE)

    if depth_buffer is None or len(depth_buffer) != NUM_RAYS:
        depth_buffer = np.full(NUM_RAYS, np.inf)
    else:
        depth_buffer[:] = np.inf

    half_h = SCREEN_HEIGHT // 2
    pitch_px = math.tan(player.pitch) * DIST_PLANE
    cam_z = (player.height - player.cam_height) + TILE / 2

    # Precompute useful scaled texture width
    tex_w = int(TILE * scale_factor)
    tex_h = tex_w  # square textures

    for ray in range(NUM_RAYS):
        dx = c * COS_REL[ray] - a * SIN_REL[ray]
        dy = a * COS_REL[ray] + c * SIN_REL[ray]

        mx, my = ptx, pty
        dxr = abs(TILE / dx) if dx else float('inf')
        dyr = abs(TILE / dy) if dy else float('inf')

        step_x, side_x = ((-1, (px - mx * TILE) * dxr / TILE) if dx < 0 else (1, ((mx + 1) * TILE - px) * dxr / TILE))
        step_y, side_y = ((-1, (py - my * TILE) * dyr / TILE) if dy < 0 else (1, ((my + 1) * TILE - py) * dyr / TILE))

        side = 0
        hits = []

        while True:
            if abs(mx - ptx) > CHUNK_RADIUS_TILES or abs(my - pty) > CHUNK_RADIUS_TILES:
                break

            stacked = []
            if USE_FILE_MAP:
                if 0 <= my < len(LOADED_MAP[0]) and 0 <= mx < len(LOADED_MAP[0][0]):
                    for z in range(len(LOADED_MAP)):
                        t = LOADED_MAP[z][my][mx]
                        if t:
                            stacked.append(t)
            else:
                t = get_tile(mx, my, 0)
                if t:
                    stacked = [t]

            if stacked:
                # Perpendicular length
                rl = ((mx - px / TILE + (1 - step_x) / 2) * TILE / dx) if side == 0 else ((my - py / TILE + (1 - step_y) / 2) * TILE / dy)
                hits.append({'distance': rl, 'stacked_tiles': stacked.copy(), 'side': side, 'dir_x': dx, 'dir_y': dy})

            if side_x < side_y:
                side_x += dxr
                mx += step_x
                side = 0
            else:
                side_y += dyr
                my += step_y
                side = 1

            if abs(mx - px / TILE) > MAX_DEPTH or abs(my - py / TILE) > MAX_DEPTH:
                break

        if not hits:
            continue
        hits.sort(key=lambda h: h['distance'], reverse=True)

        for h in hits:
            rl, tiles, s, dx_, dy_ = h['distance'], h['stacked_tiles'], h['side'], h['dir_x'], h['dir_y']

            # Calculate texture x coordinate scaled
            wall_x = (py / TILE + rl * dy_ / TILE) if s == 0 else (px / TILE + rl * dx_ / TILE)
            wall_x -= math.floor(wall_x)
            tx = int(max(0, min(wall_x * tex_w, tex_w - 1)))
            if (s == 0 and dx_ > 0) or (s == 1 and dy_ < 0):
                tx = tex_w - tx - 1

            cos_rel = dx_ * c + dy_ * a
            perp = max(0.0001, rl * cos_rel)
            wall_h = len(tiles)
            line_h = int(wall_h * tex_h * DIST_PLANE / perp)

            horizon = half_h + pitch_px
            bottom_y = horizon + (cam_z * DIST_PLANE) / perp
            top_y = bottom_y - line_h

            slice_w = 1
            slice_h_f = line_h / wall_h
            accum_y = bottom_y

            # Draw each tile slice, scaled with cached texture columns, no extra alpha blending
            for tid in tiles:
                slice_h = round(slice_h_f)
                accum_y -= slice_h
                col = col_cache.get(tid, col_cache[0])[tx]
                # We can skip the alpha darkening step here to speed things up

                # Just blit scaled column slice directly
                slice_col = pygame.transform.scale(col, (slice_w, slice_h))
                screen.blit(slice_col, (ray, int(accum_y)))

            depth_buffer[ray] = min(depth_buffer[ray], perp)

# ── fixed floor renderer ──────────────────────────────────────────────────────
def draw_floor(screen, player, floor_tex_arr, scale_factor):
    import numpy as np
    import math
    from pygame.surfarray import pixels3d

    buf = pixels3d(screen)
    half_h = SCREEN_HEIGHT // 2
    pitch_px = math.tan(player.pitch) * DIST_PLANE
    horizon = half_h + pitch_px

    # Clear ceiling with sky blue
    ceil_end = int(min(max(horizon, 0), SCREEN_HEIGHT))
    if ceil_end > 0:
        buf[:, :ceil_end] = (0, 150, 220)

    # Camera plane and directions
    cam_x = np.linspace(-1.0, 1.0, SCREEN_WIDTH)
    dir_x = math.cos(player.angle)
    dir_y = math.sin(player.angle)
    plane_x = -dir_y * math.tan(current_fov / 2)
    plane_y = dir_x * math.tan(current_fov / 2)
    ray_x = dir_x + plane_x * cam_x
    ray_y = dir_y + plane_y * cam_x

    cam_z = (player.height - player.cam_height) + TILE / 2
    tex_size = floor_tex_arr.shape[0]

    # Precompute some shading parameters
    ambient = 100
    shade_levels = np.array([0.5, 0.7, 0.85, 1.0])

    # Implement LOD: sample every N rows based on distance
    # Closer rows: full resolution, farther rows: skip rows (lower vertical res)
    max_lod_step = 4  # max step for rows, tune for quality/speed tradeoff
    y_range = np.arange(ceil_end, SCREEN_HEIGHT)

    # We will batch process rows in chunks with increasing step size (LOD)
    lod_steps = [1, 2, 4]  # steps for near, mid, far
    lod_limits = [SCREEN_HEIGHT - 60, SCREEN_HEIGHT - 120]  # change step after these rows

    # Pre-allocate floor row buffer for entire width
    floor_row = np.zeros((SCREEN_WIDTH, 3), dtype=np.float32)

    # Process rows with variable step (LOD)
    y_idx = ceil_end
    while y_idx < SCREEN_HEIGHT:
        # Determine LOD step
        if y_idx < lod_limits[1]:
            if y_idx < lod_limits[0]:
                step = lod_steps[0]  # nearest rows full res
            else:
                step = lod_steps[1]
        else:
            step = lod_steps[2]

        # Compute arrays for this batch of rows
        ys = np.arange(y_idx, min(y_idx + step, SCREEN_HEIGHT))

        # Vectorize row_dist for each y in batch
        scr_rel = ys[:, None] - horizon
        scr_rel = np.where(scr_rel == 0, 1, scr_rel)  # avoid div by zero

        row_dist = (cam_z * DIST_PLANE) / scr_rel  # shape (step,1)

        # world_x/y shape (step, width)
        world_x = player.x + row_dist * ray_x
        world_y = player.y + row_dist * ray_y

        # Texture coordinates mod texture size, floor to int for indexing
        tx = np.floor(world_x).astype(int) % tex_size
        ty = np.floor(world_y).astype(int) % tex_size

        # Sample texture for all pixels in this batch
        texels = floor_tex_arr[tx, ty]  # shape (step, width, 3)

        # Calculate continuous shade vectorized
        cont_shade = np.clip(6 / (row_dist + 1), 0.5, 1.0)  # shape (step,1)

        # Find closest shade level index (vectorized)
        diffs = np.abs(shade_levels[:, None] - cont_shade.T)  # (4, step)
        idxs = diffs.argmin(axis=0)  # shape (step,)

        # Gather shade per row
        shade = shade_levels[idxs][:, None]  # shape (step,1)

        # Blend with ambient
        shaded_tex = texels * shade[:, None] + ambient * (1 - shade)[:, None]

        # Clip and convert
        shaded_tex = np.clip(shaded_tex, 0, 255).astype(np.uint8)

        # Write each processed row back to buffer
        for i, yv in enumerate(ys):
            buf[:, yv] = shaded_tex[i]

        y_idx += step

    del buf

def draw_scanlines(surface, base_alpha=40, spacing=2, time_offset=0):
    scanline_surf = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    w, h = surface.get_size()
    
    # Time-based alpha pulsing (0.5 to 1.0 multiplier)
    pulse = (math.sin(time.time() * 3 + time_offset) + 1) / 4 + 0.5
    alpha = int(base_alpha * pulse)
    
    for y in range(0, h, spacing):
        # Horizontal sine wave shift for scanline wobble
        #x_offset = int(5 * math.sin((y / 10) + time.time() * 5))
        x_offset = 0
        pygame.draw.line(scanline_surf, (0, 0, 0, alpha), (x_offset, y), (w + x_offset, y))
    
    surface.blit(scanline_surf, (0, 0))

def find_empty_spawn(layer=0):
    if USE_FILE_MAP and LOADED_MAP:
        h, w = len(LOADED_MAP[layer]), len(LOADED_MAP[layer][0])
        cx, cy = w // 2, h // 2
        for radius in range(max(cx, cy) + 1):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    x, y = cx + dx, cy + dy
                    if 0 <= x < w and 0 <= y < h and get_tile(x, y, layer) == 0:
                        return x * TILE + TILE/2, y * TILE + TILE/2
    else:
        for radius in range(9):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    x, y = 8 + dx, 8 + dy
                    if 0 <= x < 16 and 0 <= y < 16 and get_tile(x, y, layer) == 0:
                        return x * TILE + TILE/2, y * TILE + TILE/2
    return TILE * 1.5, TILE * 1.5

MAX_NPC_RENDER_DISTANCE = 300  # Adjust as needed (in game units)

# ── fixed NPC sprite projection ───────────────────────────────────────────────
def render_npcs(screen, player, npcs):
    global depth_buffer

    px, py = player.x, player.y

    half_h   = SCREEN_HEIGHT // 2
    pitch_px = math.tan(player.pitch) * DIST_PLANE
    horizon  = half_h + pitch_px                       # shared anchor
    cam_z    = (player.height - player.cam_height) + TILE / 2

    for npc in npcs:
        dx, dy = npc.x - px, npc.y - py
        dist   = math.hypot(dx, dy)

        # Skip rendering if too far away
        if dist > MAX_NPC_RENDER_DISTANCE:
            continue

        if dist < 0.001:
            continue

        angle_to_npc = math.atan2(dy, dx)
        rel_angle    = (angle_to_npc - player.angle + math.tau) % math.tau
        if rel_angle > math.pi:
            rel_angle -= math.tau
        if abs(rel_angle) > current_fov / 2:
            continue

        perp_dist = dist * math.cos(rel_angle)         # perpendicular slice depth
        if perp_dist <= 0.001:
            continue

        sprite_w = sprite_h = int(TILE * NPC_SCALE * DIST_PLANE / perp_dist)
        screen_x = SCREEN_WIDTH / 2 + math.tan(rel_angle) * DIST_PLANE - sprite_w / 2

        screen_col = int(screen_x + sprite_w / 2)
        if (depth_buffer is not None and
                0 <= screen_col < NUM_RAYS and
                perp_dist > depth_buffer[screen_col] + 5):
            continue

        bottom_y = horizon + (cam_z * DIST_PLANE) / perp_dist
        screen_y = int(bottom_y - sprite_h)

        scaled = pygame.transform.scale(npc.texture, (sprite_w, sprite_h))
        screen.blit(scaled, (screen_x, screen_y))

def render_dialogue_box(screen, text, input_mode=_A, input_text='', scroll_offset=0):
    global FONT
    box_height = 57  # smaller height
    box = pygame.Surface((SCREEN_WIDTH, box_height))
    box.set_alpha(255 if input_mode else 220)
    box.fill((30, 30, 30))
    screen.blit(box, (0, SCREEN_HEIGHT - box_height))
    
    if FONT is _C:
        FONT = pygame.font.SysFont('Courier New', 10, bold=_B)  # smaller font
    
    words = text.split()
    lines, line = [], ''
    for w in words:
        test_line = line + w + ' '
        if FONT.size(test_line)[0] > SCREEN_WIDTH - 20:
            lines.append(line)
            line = w + ' '
        else:
            line = test_line
    if line:
        lines.append(line)
    
    line_height = FONT.get_height() + 2
    max_lines = (box_height - 20) // line_height
    max_scroll = max(0, len(lines) - max_lines)
    scroll_offset = max(0, min(scroll_offset, max_scroll))
    
    y = SCREEN_HEIGHT - box_height + 10
    for line in lines[scroll_offset:scroll_offset + max_lines]:
        screen.blit(FONT.render(line.strip(), _B, (255, 255, 255)), (10, y))
        y += line_height
    
    if input_mode:
        input_y = SCREEN_HEIGHT - 20
        screen.blit(FONT.render('> ' + input_text, _B, (180, 220, 255)), (10, input_y))
    
    return max_scroll

current_fov = FOV_DEFAULT
def update_fov(target_fov, dt, player=None):
    global current_fov, DIST_PLANE, COS_REL, SIN_REL, NUM_RAYS
    old_dist_plane = DIST_PLANE
    diff = target_fov - current_fov
    step = ZOOM_SPEED * dt
    if abs(diff) < step:
        current_fov = target_fov
    else:
        current_fov += step if diff > 0 else -step
    half_fov = current_fov / 2
    DIST_PLANE = SCREEN_WIDTH / 2 / math.tan(half_fov)
    a = np.arctan(np.linspace(-1, 1, NUM_RAYS) * math.tan(half_fov))
    COS_REL, SIN_REL = np.cos(a), np.sin(a)

    # Adjust player.pitch to keep vertical look stable if player given
    if player is not None and old_dist_plane != 0:
        # Preserve pitch offset in screen space:
        pitch_offset = math.tan(player.pitch) * old_dist_plane
        # New pitch to match old pitch offset at new DIST_PLANE:
        player.pitch = math.atan(pitch_offset / DIST_PLANE)

item_textures = ['notebook', 'cat']

def load_item_textures():
    items = {}
    # Example: load a couple of item textures
    for item_texture in item_textures:
        items[item_texture] = pygame.image.load(f"{textures_folder}/{item_texture}.png").convert_alpha()
    return items

def render_inventory_hud(screen, player, item_textures, pos=(10, 10), slot_size=40, padding=5, font=None):
    x, y = pos
    inv = player.inventory
    for i, item_key in enumerate(inv):
        icon = item_textures.get(item_key)
        if icon:
            # Scale icon to slot_size (keep aspect ratio)
            icon_surf = pygame.transform.smoothscale(icon, (slot_size, slot_size))
            # Draw background rectangle for slot
            rect = pygame.Rect(x + i * (slot_size + padding), y, slot_size, slot_size)
            pygame.draw.rect(screen, (50, 50, 50), rect)
            # Highlight current selection
            if i == player.inventory_index:
                pygame.draw.rect(screen, (255, 255, 0), rect, 3)
            else:
                pygame.draw.rect(screen, (150, 150, 150), rect, 1)
            # Blit the icon
            screen.blit(icon_surf, (rect.x, rect.y))
            # Optionally draw item name below icon
            if font:
                name_surf = font.render(item_key, True, (255, 255, 255))
                name_rect = name_surf.get_rect(center=(rect.centerx, rect.bottom + 12))
                screen.blit(name_surf, name_rect)

def play_music(path, loop=True):
    try:
        pygame.mixer.music.load(path)
        pygame.mixer.music.play(-1 if loop else 0)
        print(f"Playing music: {path}")
    except Exception as e:
        print(f"Failed to play music '{path}': {e}")

def run_intro(screen, lore_text):
    play_music(os.path.join("sounds", "intro_music.mp3"), loop=True)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('Courier New', 14, bold=True)
    width, height = screen.get_size()
    lines = []

    # Split lore into lines that fit the screen width
    words = lore_text.split()
    line = ''
    for w in words:
        test_line = line + w + ' '
        if font.size(test_line)[0] > width - 40:
            lines.append(line)
            line = w + ' '
        else:
            line = test_line
    if line:
        lines.append(line)

    # Rolling variables
    current_line = 0
    char_index = 0
    finished = False
    skip = False
    intro_text = ''

    while not finished and not skip:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_RETURN, pygame.K_SPACE, pygame.K_ESCAPE):
                    skip = True  # skip the intro on key press

        screen.fill((0, 0, 0))

        # Build the currently visible text (character rolling)
        if current_line < len(lines):
            if char_index < len(lines[current_line]):
                char_index += 1
            else:
                # pause briefly on full line, then go to next line
                pygame.time.wait(500)
                current_line += 1
                char_index = 0
        else:
            finished = True

        # Compose text to render: all fully displayed lines + partial current line
        visible_text = '\n'.join(lines[:current_line])
        if current_line < len(lines):
            visible_text += '\n' + lines[current_line][:char_index]

        # Render text lines with some vertical spacing
        y = height // 4
        for render_line in visible_text.split('\n'):
            text_surf = font.render(render_line, True, (255, 255, 255))
            screen.blit(text_surf, (20, y))
            y += font.get_height() + 5

        # Instruction prompt
        prompt = font.render("Press Enter or Space to skip...", True, (180, 180, 180))
        screen.blit(prompt, (width - prompt.get_width() - 20, height - 40))

        pygame.display.flip()
        clock.tick(30)

    pygame.mixer.music.stop()

def main():
    global SCREEN_WIDTH, SCREEN_HEIGHT, show_fps, FONT, depth_buffer
    global dialogue_active, input_active, input_text, ai_response, scroll_offset
    bob_timer = 0.0
    pygame.init()
    pygame.mixer.init()
    pygame.mouse.set_visible(_A)
    pygame.event.set_grab(_B)
    load_ussa_map('city_map.ussa3d')
    info = pygame.display.Info()
    flags = pygame.FULLSCREEN | pygame.SCALED if FULLSCREEN else 0
    disp = pygame.display.set_mode((info.current_w, info.current_h), flags)
    SCREEN_WIDTH, SCREEN_HEIGHT = int(info.current_w * RENDER_SCALE), int(info.current_h * RENDER_SCALE)
    update_ray_params(SCREEN_WIDTH)
    gs = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
    item_textures = load_item_textures()
    run_intro(disp, intro_lore)
    play_music(os.path.join("sounds", "game_music.mp3"), loop=True)
    p = Player()
    p.inventory = ['notebook']
    p.held_item = item_textures.get('notebook')  # or None if no item held
    p.select_item(0, item_textures)       # Start holding first item
    p.x, p.y = find_empty_spawn(p.layer)
    textures, col_cache, floor_arr, scale_factor = load_textures()
    npc_textures = load_npc_textures()
    item_textures = load_item_textures()

    print(item_textures.keys())  # Check what keys are loaded
    print(p.inventory)           # Check what keys you put in inventory

    # --- Even NPC spawning throughout the map ---
    npcs = []
    if USE_FILE_MAP and LOADED_MAP:
        layer = 0
        map_height = len(LOADED_MAP[layer])
        map_width = len(LOADED_MAP[layer][0])
    else:
        map_width, map_height = 16, 16

    num_npcs = 4000
    grid_cols = int(math.sqrt(num_npcs * map_width / map_height))
    grid_rows = int(num_npcs / grid_cols) + 1

    x_step = map_width / grid_cols
    y_step = map_height / grid_rows

    placed = 0
    for row in range(grid_rows):
        for col in range(grid_cols):
            if placed >= num_npcs:
                break
            tx = int(col * x_step + x_step / 2)
            ty = int(row * y_step + y_step / 2)
            tx = max(0, min(map_width - 1, tx))
            ty = max(0, min(map_height - 1, ty))

            spawn_x, spawn_y = None, None
            found = False
            for radius in range(3):
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        nx, ny = tx + dx, ty + dy
                        if 0 <= nx < map_width and 0 <= ny < map_height:
                            if get_tile(nx, ny, 0) == 0:
                                spawn_x = nx * TILE + TILE / 2
                                spawn_y = ny * TILE + TILE / 2
                                found = True
                                break
                    if found:
                        break
                if found:
                    break

            if found:
                tex = random.choice(npc_textures)
                npcs.append(NPC(spawn_x, spawn_y, tex))
                placed += 1

    ww, wh = disp.get_size()
    scale = min(ww // SCREEN_WIDTH, wh // SCREEN_HEIGHT)
    sw, sh = SCREEN_WIDTH * scale, SCREEN_HEIGHT * scale
    ox, oy = (ww - sw) // 2, (wh - sh) // 2
    clock = pygame.time.Clock()
    MS_X, MS_Y = .002, .002
    depth_buffer = np.full(NUM_RAYS, np.inf)
    esc_pressed = _A
    dialogue_active = _A
    input_active = _A
    input_text = ''
    ai_response = ''
    scroll_offset = 0

    while _B:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    if dialogue_active:
                        dialogue_active = _A
                        input_active = _A
                        input_text = ''
                        ai_response = ''
                        scroll_offset = 0
                    elif esc_pressed:
                        pygame.quit()
                        sys.exit()
                    else:
                        esc_pressed = _B
                        print('Press ESC again to quit')
                    continue
                else:
                    esc_pressed = _A
                if e.key == pygame.K_p:
                    import datetime
                    fn = f"screenshot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    pygame.image.save(gs, fn)
                    print(f"Screenshot saved as {fn}")
                elif e.key == pygame.K_PAGEUP and p.layer < len(LOADED_MAP) - 1:
                    p.layer += 1
                    print(f"Switched to layer {p.layer}")
                elif e.key == pygame.K_PAGEDOWN and p.layer > 0:
                    p.layer -= 1
                    print(f"Switched to layer {p.layer}")
                elif e.key == pygame.K_TAB:
                    if p.inventory:
                        new_index = (p.inventory_index + 1) % len(p.inventory)
                        p.select_item(new_index, item_textures)
                if input_active:
                    if e.key == pygame.K_RETURN:
                        if input_text.strip():
                            user_msg = (lore + "\nAnswer extremely, EXTREMELY briefly. Don't narrate actions. Respond in character. Don't use special characters." + input_text.strip())
                            input_text = ''
                            ai_response = 'Thinking...'
                            scroll_offset = 0
                            try:
                                response = chat(model='gemma3:1b', messages=[{'role': 'user', 'content': user_msg}])
                                ai_response = response.message.content
                            except Exception as ex:
                                ai_response = f"Error: {ex}"
                    elif e.key == pygame.K_BACKSPACE:
                        input_text = input_text[:-1]
                    elif e.key == pygame.K_UP:
                        scroll_offset = max(0, scroll_offset - 1)
                    elif e.key == pygame.K_DOWN:
                        scroll_offset = min(current_max_scroll, scroll_offset + 1)
                    elif e.unicode and e.unicode.isprintable():
                        input_text += e.unicode
                    continue
                if e.key == pygame.K_e:
                    if dialogue_active:
                        dialogue_active = _A
                        input_active = _A
                        input_text = ''
                        ai_response = ''
                        scroll_offset = 0
                    else:
                        for npc in npcs:
                            if math.hypot(npc.x - p.x, npc.y - p.y) < TILE * 1.5:
                                dialogue_active = _B
                                input_active = _B
                                input_text = ''
                                ai_response = 'Hello! Ask me anything.'
                                scroll_offset = 0
                                break
                elif dialogue_active and e.key == pygame.K_UP:
                    scroll_offset = max(0, scroll_offset - 1)
                elif dialogue_active and e.key == pygame.K_DOWN:
                    scroll_offset = min(current_max_scroll, scroll_offset + 1)
                if e.key == pygame.K_f:
                    show_fps = not show_fps

        pygame.mouse.set_pos(ww // 2, wh // 2)
        dx, dy = pygame.mouse.get_rel()
        p.angle = (p.angle + dx * MS_X) % math.tau
        p.pitch = max(-math.pi / 2, min(math.pi / 2, p.pitch - dy * MS_Y))
        dt = clock.tick(60) / 1000
        fps = clock.get_fps()
        if p.is_moving:
            bob_timer += dt * 8  # speed of bobbing (adjust 8 as you like)
        else:
            # Reset bob smoothly to 0 when not moving:
            bob_timer = 0.0
        if not dialogue_active or not input_active:
            p.movement(dt)
        if not dialogue_active:
            for npc in npcs:
                npc.update(dt)
        gs.fill(0)
        draw_floor(gs, p, floor_arr, scale_factor)
        zooming = pygame.key.get_pressed()[pygame.K_c]
        update_fov(ZOOMED_FOV if zooming else FOV_DEFAULT, dt, p)
        ray_cast(gs, p, col_cache, scale_factor)
        render_npcs(gs, p, npcs)
        #dark = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        #dark.fill((10, 10, 40))
        #dark.set_alpha(150)
        #gs.blit(dark, (0, 0))
        if dialogue_active:
            current_max_scroll = render_dialogue_box(gs, ai_response or 'Hello!', input_mode=input_active, input_text=input_text, scroll_offset=scroll_offset)

        draw_scanlines(gs)

        # Scale main surface gs up to display size
        up = pygame.transform.scale(gs, (sw, sh))
        disp.fill(0)
        disp.blit(up, (ox, oy))

        # Render FPS and stats overlay at full display resolution
        if show_fps:
            if FONT is _C:
                FONT = pygame.font.SysFont('Arial', 36)  # bigger font for readability

            lines = [
                f"USSAGAME ver. 052825",
                f"FPS: {int(fps)}",
                f"Player Pos: ({int(p.x)}, {int(p.y)}), {p.layer}",
                f"Player Angle: {math.degrees(p.angle):.2f}°",
                f"Player Pitch: {math.degrees(p.pitch):.2f}°",
                f"FOV: {math.degrees(current_fov):.2f}°",
                f"Zoomed: {'Yes' if zooming else 'No'}",
                f"Dialogue Active: {'Yes' if dialogue_active else 'No'}",
                f"Input Active: {'Yes' if input_active else 'No'}",
                f"NPC Count: {len(npcs)}",
                f"Depth Buffer: {depth_buffer.min():.2f} - {depth_buffer.max():.2f}",
                f"Map Layers: {len(LOADED_MAP) if USE_FILE_MAP else 1}",
                f"Map Size: {len(LOADED_MAP[0])}x{len(LOADED_MAP[0][0])}",
                f"Propaganda Disabled: {'Yes' if disable_propoganda else 'No'}",
                f"Screen Size: {SCREEN_WIDTH}x{SCREEN_HEIGHT}",
                f"Render Scale: {RENDER_SCALE:.2f}",
                f"Fullscreen: {'Yes' if FULLSCREEN else 'No'}",
            ]
            y = 5
            fps_surface = pygame.Surface((ww, wh), pygame.SRCALPHA)
            for line in lines:
                text_surf = FONT.render(line, True, (255, 255, 0))
                fps_surface.blit(text_surf, (5, y))
                y += FONT.get_height() + 2
            disp.blit(fps_surface, (0, 130))

        if p.held_item and not dialogue_active:
            bob_amplitude = 10  # max vertical bob in pixels
            bob_offset = 0
            if p.is_moving:
                bob_offset = math.sin(bob_timer) * bob_amplitude
            # Lower base position by bob_amplitude so the item "rests" at the bottom
            held_img = pygame.transform.scale(p.held_item, (550, 550))
            padding = 0
            pos_x = ww - held_img.get_width() - padding
            pos_y = wh - held_img.get_height() - padding + bob_amplitude + bob_offset
            disp.blit(held_img, (pos_x, pos_y))

        render_inventory_hud(disp, p, item_textures, pos=(30, 30), slot_size=60, padding=24, font=pygame.font.SysFont('Courier New', 16))

        pygame.display.flip()

if __name__ == '__main__':
    main()
