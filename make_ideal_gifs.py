# make_ideal_gifs.py — v4
import os
import json
import math
import heapq
import argparse
from typing import Dict, Any, List, Tuple, Optional, Deque
from collections import deque, defaultdict

import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from environment.red_and_blue_env_cnn import RedAndBlueEnv


# ============================= Utils ==============================

def clamp(v, lo, hi): return lo if v < lo else hi if v > hi else v


# =========================== Okabe–Ito ============================

class Palette:
    # Okabe–Ito + из них производные (RGBA для прозрачностей)
    BLACK   = (0, 0, 0)
    ORANGE  = (230, 159, 0)
    SKY     = (86, 180, 233)
    BLU     = (0, 114, 178)
    VERD    = (0, 158, 115)
    YELL    = (240, 228, 66)
    RED     = (213, 94, 0)     # оранжево-красный (для агента хорошо)
    MAG     = (204, 121, 167)
    GREY    = (120, 120, 120)

class Theme:
    # фон/тайлы
    BG_FREE        = (245, 247, 249)
    OBSTACLE       = (35, 39, 42)
    VISITED        = (230, 210, 220)
    LAST_SEEN      = (175, 210, 255)
    EXP_FREE_RED   = (180, 235, 200)   # explored free by red
    EXP_FREE_BLUE  = (200, 220, 255)   # explored free by blue

    # агенты
    RED            = Palette.RED
    RED_HELPER     = (180, 70, 40)
    BLUE           = Palette.BLU
    GREY    = (120, 120, 120)

    # fov (RGBA)
    FOV_RED        = (Palette.RED[0], Palette.RED[1], Palette.RED[2], 64)
    FOV_HELPER     = (Palette.ORANGE[0], Palette.ORANGE[1], Palette.ORANGE[2], 48)
    FOV_BLUE       = (Palette.SKY[0], Palette.SKY[1], Palette.SKY[2], 40)

    # контуры FOV
    FOV_OUT_RED    = (Palette.RED[0], Palette.RED[1], Palette.RED[2], 160)
    FOV_OUT_HELPER = (Palette.ORANGE[0], Palette.ORANGE[1], Palette.ORANGE[2], 130)
    FOV_OUT_BLUE   = (Palette.SKY[0], Palette.SKY[1], Palette.SKY[2], 120)

    PATH           = Palette.VERD
    TARGET         = Palette.YELL
    GRID           = (205, 210, 215)
    TEXT           = (20, 20, 20)
    LEG_BG         = (255, 255, 255, 235)
    LEG_BORDER     = (60, 60, 60, 255)

    GOOD_LOS       = (0, 170, 90)      # LOS чистый
    BAD_LOS        = (200, 60, 60)     # LOS перекрыт


# ============================ A* path ============================

def astar(grid: np.ndarray, start: Tuple[int,int], goal: Tuple[int,int]) -> Optional[List[Tuple[int,int]]]:
    H, W = grid.shape
    if not (0 <= start[0] < H and 0 <= start[1] < W and 0 <= goal[0] < H and 0 <= goal[1] < W):
        return None
    if grid[start] == 1 or grid[goal] == 1:
        return None

    def h(a, b): return abs(a[0]-b[0]) + abs(a[1]-b[1])

    neigh = [(1,0),(-1,0),(0,1),(0,-1)]
    openh = [(h(start, goal), 0, start)]
    came  = {}
    gscore= {start:0}

    while openh:
        _, g, cur = heapq.heappop(openh)
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return list(reversed(path))
        for dx,dy in neigh:
            nx, ny = cur[0]+dx, cur[1]+dy
            if nx<0 or ny<0 or nx>=H or ny>=W or grid[nx,ny]==1: continue
            ng = g+1
            nxt = (nx, ny)
            if ng < gscore.get(nxt, 1e9):
                gscore[nxt] = ng
                came[nxt] = cur
                heapq.heappush(openh, (ng + h(nxt, goal), ng, nxt))
    return None


# =========================== Bresenham LOS ========================

def bresenham_line(a: Tuple[int,int], b: Tuple[int,int]) -> List[Tuple[int,int]]:
    (x0, y0), (x1, y1) = a, b
    dx, dy = abs(x1-x0), abs(y1-y0)
    sx = 1 if x0<x1 else -1
    sy = 1 if y0<y1 else -1
    err = dx-dy
    pts = []
    while True:
        pts.append((x0, y0))
        if x0==x1 and y0==y1: break
        e2 = 2*err
        if e2 > -dy: err -= dy; x0 += sx
        if e2 <  dx: err += dx; y0 += sy
    return pts

def los_clear(grid: np.ndarray, a: Tuple[int,int], b: Tuple[int,int]) -> bool:
    for p in bresenham_line(a, b):
        if grid[p] == 1:
            return False
    return True


# ======================== Oracle Controller ======================

def angle_to(a: Tuple[int,int], b: Tuple[int,int]) -> float:
    dx, dy = b[0]-a[0], b[1]-a[1]
    return (math.degrees(math.atan2(dy, dx)) + 360) % 360

class OracleController:
    def __init__(self, turn_interval=3, replan_every=5):
        self.turn_interval = int(turn_interval)
        self.replan_every  = int(replan_every)
        self.reset()

    def reset(self):
        self.path: List[Tuple[int,int]] = []
        self.path_idx = 0
        self.target: Optional[Tuple[int,int]] = None
        self.t = 0
        self.replans = 0
        self.last_target: Optional[Tuple[int,int]] = None

    def choose_action(self, env: RedAndBlueEnv, events: Deque[str]) -> int:
        self.t += 1
        blues = [tuple(p) for p in env.blues_pos]
        if not blues:
            return 4

        if self.target is None or self.target not in blues:
            self.target = min(blues, key=lambda b: abs(b[0]-env.red_pos[0]) + abs(b[1]-env.red_pos[1]))
            if self.last_target != self.target:
                events.appendleft("Target changed")
                self.last_target = self.target
            self._replan(env)
        if self.t % self.replan_every == 0:
            self.target = min(blues, key=lambda b: abs(b[0]-env.red_pos[0]) + abs(b[1]-env.red_pos[1]))
            self._replan(env)

        if not self.path or self.path_idx >= len(self.path):
            self._replan(env)
            return 4

        desired = angle_to(tuple(env.red_pos), self.target)
        diff = (desired - env.red_angle) % 360
        if self.t % self.turn_interval == 0 and diff > env.view_angle / 2:
            return 4

        next_cell = self.path[self.path_idx]
        if tuple(env.red_pos) == next_cell and self.path_idx + 1 < len(self.path):
            self.path_idx += 1
            next_cell = self.path[self.path_idx]

        dx, dy = next_cell[0]-env.red_pos[0], next_cell[1]-env.red_pos[1]
        if   dx==-1 and dy==0: return 0
        elif dx== 1 and dy==0: return 1
        elif dx== 0 and dy==-1: return 2
        elif dx== 0 and dy== 1: return 3
        else: return 4

    def _replan(self, env: RedAndBlueEnv):
        if self.target is None:
            self.path, self.path_idx = [], 0
            return
        p = astar(env.grid, tuple(env.red_pos), self.target)
        if not p or len(p) < 2:
            others = [tuple(p) for p in env.blues_pos if tuple(p) != self.target]
            for tgt in sorted(others, key=lambda b: abs(b[0]-env.red_pos[0]) + abs(b[1]-env.red_pos[1])):
                p2 = astar(env.grid, tuple(env.red_pos), tgt)
                if p2 and len(p2) >= 2:
                    self.target = tgt
                    p = p2
                    break
        self.path = p if p else []
        self.path_idx = 1 if self.path else 0
        self.replans += 1


# ======================== Drawing helpers ========================

def draw_grid(d: ImageDraw.ImageDraw, H: int, W: int, scale: int, every: int):
    if every <= 0: return
    # адаптивная тонкость
    col = Theme.GRID
    width = 1 if scale >= 8 else 0  # на маленьких масштабах отключаем
    if width == 0: return
    for i in range(0, H, every):
        y = i * scale
        d.line([(0, y), (W*scale, y)], fill=col, width=1)
    for j in range(0, W, every):
        x = j * scale
        d.line([(x, 0), (x, H*scale)], fill=col, width=1)

def ellipse_clamped(d: ImageDraw.ImageDraw, x, y, scale, r_in, color):
    x0, y0 = y*scale + r_in, x*scale + r_in
    x1, y1 = (y+1)*scale - 1 - r_in, (x+1)*scale - 1 - r_in
    if x1 < x0: x1 = x0
    if y1 < y0: y1 = y0
    d.ellipse([x0, y0, x1, y1], fill=color)

def draw_legend(canvas: Image.Image, items: list[tuple[str, tuple]], scale: int = 8) -> None:
    """
    Рисует легенду в 2 строки (горизонтальное расположение).
    items: [(label, color_or_rgba), ...]
    """
    assert canvas.mode == "RGBA"
    W, H = canvas.size
    draw = ImageDraw.Draw(canvas, "RGBA")

    # размеры
    pad = max(6, scale)
    hgap = max(10, scale // 2)
    vgap = max(6, scale // 2)
    sw = max(10, scale + 4)
    sh = sw

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 10 + scale // 2)
    except:
        font = ImageFont.load_default()
    text_h = font.getbbox("Hg")[3] - font.getbbox("Hg")[1]

    # делим элементы на 2 строки
    half = (len(items) + 1) // 2
    rows = [items[:half], items[half:]]

    y_cursor = pad
    for row in rows:
        x_cursor = pad
        for label, color in row:
            # нормализуем цвет
            if isinstance(color, tuple):
                if len(color) == 3:
                    rgba = (*color, 255)
                elif len(color) == 4:
                    rgba = color
                else:
                    rgba = (0, 0, 0, 0)
            else:
                rgba = (0, 0, 0, 0)

            # special LOS
            if label.strip().lower().startswith("los:"):
                x0, y0 = x_cursor, y_cursor
                draw.rectangle([x0, y0, x0 + sw, y0 + sh], outline=(150, 150, 150, 160), width=1)
                draw.line([x0+2, y0+sh-2, x0+sw-2, y0+2], fill=(0,170,90,230), width=2)
                draw.line([x0+2, y0+2, x0+sw-2, y0+sh-2], fill=(200,60,60,230), width=2)
                tx = x0 + sw + hgap
                draw.text((tx, y0), label, fill=(20,20,20,255), font=font)
                x_cursor = tx + draw.textlength(label, font=font) + hgap
            else:
                x0, y0 = x_cursor, y_cursor
                draw.rectangle([x0, y0, x0+sw, y0+sh], fill=rgba, outline=(120,120,120,160), width=1)
                tx = x0 + sw + hgap
                draw.text((tx, y0), label, fill=(20,20,20,255), font=font)
                x_cursor = tx + draw.textlength(label, font=font) + hgap

        y_cursor += sh + vgap


def build_legend_image(total_width: int, legend_lines, scale_for_legend: int) -> Image.Image:
    """
    Рендерит legend в «большое» полотно и затем автокропит по непустой области.
    Возвращает RGBA-изображение легенды шириной не больше total_width.
    """
    # Большое полотно: высота с запасом (400 px)
    canvas = Image.new("RGBA", (total_width, 400), (0, 0, 0, 0))
    draw_legend(canvas, legend_lines, scale_for_legend)
    bbox = canvas.getbbox()
    if bbox:
        legend = canvas.crop(bbox)
    else:
        legend = canvas
    return legend


def compose_with_legend_bottom(top_image: Image.Image, legend_img: Image.Image, margin: int = 6) -> Image.Image:
    """
    Склеивает основной кадр и легенду снизу по центру с отступом.
    """
    W = top_image.width
    # не растягиваем легенду шире кадра
    if legend_img.width > W:
        legend_img = legend_img.resize((W, int(legend_img.height * (W / legend_img.width))), Image.NEAREST)
    out = Image.new("RGBA", (W, top_image.height + legend_img.height + margin), Theme.BG_FREE + (255,))
    out.alpha_composite(top_image, (0, 0))
    x_legend = (W - legend_img.width) // 2
    out.alpha_composite(legend_img, (x_legend, top_image.height + margin))
    return out


# ВСТАВЬ ЭТО ВМЕСТО ТЕКУЩЕГО stroke_circle (и рядом добавь safe_ellipse)

def safe_ellipse(draw: ImageDraw.ImageDraw, box, *, fill=None, outline=None, width=1):
    """
    Безопасная отрисовка эллипса: гарантирует x1>=x0, y1>=y0.
    Если не помещается (ноль/отрицательная ширина/высота) — превращает в точку/линию.
    """
    x0, y0, x1, y1 = map(int, box)
    if x1 < x0: x1 = x0
    if y1 < y0: y1 = y0
    # Если совсем точка — рисуем маленький прямоугольник вместо эллипса
    if x1 == x0 and y1 == y0:
        draw.rectangle([x0, y0, x1+1, y1+1], fill=fill or outline or (255, 255, 255, 255))
        return
    draw.ellipse([x0, y0, x1, y1], fill=fill, outline=outline, width=width)

def stroke_circle(img: Image.Image, x, y, scale, r_in, fill,
                  outline=(255, 255, 255, 180), outline2=(0, 0, 0, 120)):
    """
    Кружок с двойной обводкой + заливка.
    r_in автоматически зажимается так, чтобы все три слоя гарантированно влезали.
    """
    # максимальный радиус, чтобы внутренний слой (±2) не схлопнулся
    # доступный полурадиус клетки в пикселях:
    half = (scale - 1) // 2
    # три слоя: внешний контур (+1), основной контур (0), заливка (-2)
    # значит базовый r_in должен быть <= half - 2
    r_max = max(0, half - 2)
    r = int(max(0, min(r_in, r_max)))

    x0 = y * scale + r
    y0 = x * scale + r
    x1 = (y + 1) * scale - 1 - r
    y1 = (x + 1) * scale - 1 - r

    d = ImageDraw.Draw(img, "RGBA")

    # внешний «тёмный» обвод
    safe_ellipse(d, [x0 - 1, y0 - 1, x1 + 1, y1 + 1], outline=outline2, width=2)
    # светлый обвод
    safe_ellipse(d, [x0, y0, x1, y1], outline=outline, width=2)
    # заливка (чуть меньше, чтобы не перекрывать обводку)
    safe_ellipse(d, [x0 + 2, y0 + 2, x1 - 2, y1 - 2], fill=fill)


def draw_arrow(d: ImageDraw.ImageDraw, cell: Tuple[int,int], angle_deg: float, scale: int, color, length_cells=3.2, width=2):
    x, y = cell
    cx = y*scale + scale//2
    cy = x*scale + scale//2
    tx = cx + int(round(length_cells*scale*math.cos(math.radians(angle_deg))))
    ty = cy + int(round(length_cells*scale*math.sin(math.radians(angle_deg))))
    d.line([(cx, cy), (tx, ty)], fill=color, width=width)

def draw_sector_alpha(base_rgba: Image.Image, center: Tuple[int,int], view_angle: float, direction_deg: float,
                      radius: int, color_rgba: Tuple[int,int,int,int], scale: int, outline_rgba=None):
    overlay = Image.new("RGBA", base_rgba.size, (0,0,0,0))
    d = ImageDraw.Draw(overlay, "RGBA")
    cx, cy = center
    cxp, cyp = cy*scale + scale//2, cx*scale + scale//2
    pts = [(cxp, cyp)]
    step_deg = max(1, int(90 / max(10, radius)))
    half = view_angle / 2
    for da in np.arange(-half, half+0.1, step_deg):
        ang = (direction_deg + da) % 360
        px = cxp + int(round(radius*scale*math.cos(math.radians(ang))))
        py = cyp + int(round(radius*scale*math.sin(math.radians(ang))))
        pts.append((px, py))
    d.polygon(pts, fill=color_rgba)
    # контур FOV
    if outline_rgba is not None:
        arc_pts = []
        for da in np.arange(-half, half+0.1, max(1, step_deg//2)):
            ang = (direction_deg + da) % 360
            px = cxp + int(round(radius*scale*math.cos(math.radians(ang))))
            py = cyp + int(round(radius*scale*math.sin(math.radians(ang))))
            arc_pts.append((px, py))
        if len(arc_pts) >= 2:
            d.line(arc_pts, fill=outline_rgba, width=max(2, scale//3))
        d.line([(cxp, cyp), arc_pts[0]], fill=outline_rgba, width=max(2, scale//3))
        d.line([(cxp, cyp), arc_pts[-1]], fill=outline_rgba, width=max(2, scale//3))
    base_rgba.alpha_composite(overlay)


# ============================ Main render =========================

def simulate_ideal_gif(
    scen_name: str,
    env_kwargs: Dict[str, Any],
    out_dir: str,
    steps_limit: int = 800,
    seed: int = 12345,
    scale: int = 8,
    mini_scale: int = 4,
    fps: int = 18,
    show_grid_every: int = 8,
    draw_fovs: bool = True,
    send_tg: bool = False,  # зарезервировано
) -> str:
    """
    Рендер идеального ролика:
      - основная карта слева (FOV с контурами, LOS, путь A*, глоу-точки, стрелки)
      - правая колонка: Red/Blue/Shared knowledge мини-карты + HUD + timeline
      - легенда внизу (не перекрывает карту)
      - Shared Knowledge автоматически ужимается, если не помещается по высоте.
    """
    # ---------- локальные хелперы для легенды снизу ----------
    def build_legend_image(total_width: int, legend_lines, scale_for_legend: int) -> Image.Image:
        canvas = Image.new("RGBA", (total_width, 400), (0, 0, 0, 0))
        draw_legend(canvas, legend_lines, scale_for_legend)
        bbox = canvas.getbbox()
        return canvas.crop(bbox) if bbox else canvas

    def compose_with_legend_bottom(top_image: Image.Image, legend_img: Image.Image, margin: int = 6) -> Image.Image:
        W = top_image.width
        if legend_img.width > W:
            legend_img = legend_img.resize((W, int(legend_img.height * (W / legend_img.width))), Image.NEAREST)
        out = Image.new("RGBA", (W, top_image.height + legend_img.height + margin), Theme.BG_FREE + (255,))
        out.alpha_composite(top_image, (0, 0))
        x_legend = (W - legend_img.width) // 2
        out.alpha_composite(legend_img, (x_legend, top_image.height + margin))
        return out
    # ----------------------------------------------------------

    os.makedirs(out_dir, exist_ok=True)

    env_kwargs = dict(env_kwargs)
    env_kwargs.setdefault("use_helpers", True)
    env_kwargs.setdefault("num_helpers", 2)
    env_kwargs.setdefault("use_last_seen", True)
    env_kwargs.setdefault("use_obstacle_memory", True)
    env_kwargs.setdefault("use_detection_timer", True)

    env = RedAndBlueEnv(**env_kwargs, seed=seed)
    obs, info = env.reset()

    ctrl = OracleController(turn_interval=3, replan_every=5)
    ctrl.reset()

    H = W = env.grid_size

    # знания/теплота
    red_known_obst  = np.zeros((H, W), dtype=np.uint8)
    blue_known_obst = np.zeros((H, W), dtype=np.uint8)
    red_known_free  = np.zeros((H, W), dtype=np.uint8)
    blue_known_free = np.zeros((H, W), dtype=np.uint8)
    visit_count     = np.zeros((H, W), dtype=np.uint16)

    newly_known_red  = np.zeros((H, W), dtype=np.uint8)
    newly_known_blue = np.zeros((H, W), dtype=np.uint8)

    events: Deque[str] = deque(maxlen=6)
    prev_dist = None
    los_green_ratio = 0
    los_samples = 0

    # геометрия кадра
    main_w, main_h = W*scale, H*scale
    side_w = max(W*mini_scale, 340)
    margin = 12
    frame_w = main_w + margin + side_w
    frame_h = main_h

    frames: List[np.ndarray] = []
    terminated = truncated = False
    t = 0
    seen_any_blue_this_ep = 0

    # FOV дискретизация
    def fov_cells(cx:int, cy:int, ang:float, view_dist:int, view_angle:float):
        cells = []
        R = view_dist
        for i in range(-R, R+1):
            xi = cx + i
            if xi < 0 or xi >= H: continue
            for j in range(-R, R+1):
                yj = cy + j
                if yj < 0 or yj >= W: continue
                if i*i + j*j > R*R: continue
                a_to = (math.degrees(math.atan2(j, i)) + 360) % 360
                diff = abs((ang - a_to + 360) % 360)
                if diff <= view_angle/2:
                    cells.append((xi, yj))
        return cells

    while not (terminated or truncated) and t < steps_limit:
        # ----- метрики -----
        if ctrl.target is not None:
            cur_dist = float(np.hypot(ctrl.target[0]-env.red_pos[0], ctrl.target[1]-env.red_pos[1]))
            delta_d  = 0.0 if prev_dist is None else (prev_dist - cur_dist)
            prev_dist = cur_dist
        else:
            cur_dist, delta_d = float('nan'), 0.0

        # ----- знания по FOV -----
        newly_known_red[:] = 0
        newly_known_blue[:] = 0

        viewers_red = [(tuple(env.red_pos), env.red_angle)]
        for (hx, hy), hang in zip(env.helpers_pos, getattr(env, "helpers_angle", [])):
            viewers_red.append(((hx, hy), hang))

        # red knowledge
        for (x, y), ang in viewers_red:
            for cx, cy in fov_cells(x, y, ang, env.view_distance, env.view_angle):
                visit_count[cx, cy] += 1
                if env.grid[cx, cy] == 1:
                    if red_known_obst[cx, cy] == 0:
                        red_known_obst[cx, cy] = 1
                        newly_known_red[cx, cy] = 1
                else:
                    if red_known_free[cx, cy] == 0:
                        red_known_free[cx, cy] = 1
                        newly_known_red[cx, cy] = 1

        # blue knowledge
        for (bx, by), bang in zip(env.blues_pos, env.blues_angle):
            for cx, cy in fov_cells(bx, by, bang, env.view_distance, env.view_angle):
                if env.grid[cx, cy] == 1:
                    if blue_known_obst[cx, cy] == 0:
                        blue_known_obst[cx, cy] = 1
                        newly_known_blue[cx, cy] = 1
                else:
                    if blue_known_free[cx, cy] == 0:
                        blue_known_free[cx, cy] = 1
                        newly_known_blue[cx, cy] = 1

        # ---------- основной холст ----------
        full = Image.new("RGBA", (frame_w, frame_h), Theme.BG_FREE + (255,))
        d_full = ImageDraw.Draw(full, "RGBA")

        # ----- основная карта -----
        base = Image.new("RGB", (main_w, main_h), Theme.BG_FREE)
        d = ImageDraw.Draw(base)

        # heatmap посещений
        max_vis = max(1, int(visit_count.max()))
        if max_vis > 0:
            irgba = ImageDraw.Draw(base, "RGBA")
            for x, y in np.argwhere(visit_count > 0):
                a = int(max(30, min(230, 30 + 200 * (visit_count[x, y] / max_vis))))
                irgba.rectangle([y*scale, x*scale, (y+1)*scale-1, (x+1)*scale-1],
                                fill=(Theme.VISITED[0], Theme.VISITED[1], Theme.VISITED[2], a))

        # last_seen (если есть)
        if getattr(env, "last_seen_blue", None) is not None and env_kwargs.get("use_last_seen", True):
            irgba = ImageDraw.Draw(base, "RGBA")
            for x, y in np.argwhere(env.last_seen_blue):
                irgba.rectangle([y*scale, x*scale, (y+1)*scale-1, (x+1)*scale-1],
                                fill=(Theme.LAST_SEEN[0], Theme.LAST_SEEN[1], Theme.LAST_SEEN[2], 95))

        # препятствия
        for x, y in np.argwhere(env.grid == 1):
            d.rectangle([y*scale, x*scale, (y+1)*scale-1, (x+1)*scale-1], fill=Theme.OBSTACLE)

        # сетка
        draw_grid(d, H, W, scale, every=show_grid_every)

        frame = base.convert("RGBA")

        # FOV на основной карте
        if draw_fovs:
            for (bx, by), bang in zip(env.blues_pos, env.blues_angle):
                draw_sector_alpha(frame, (bx, by), env.view_angle, bang, env.view_distance,
                                  Theme.FOV_BLUE, scale, outline_rgba=Theme.FOV_OUT_BLUE)
            for (hx, hy), hang in zip(env.helpers_pos, getattr(env, "helpers_angle", [])):
                draw_sector_alpha(frame, (hx, hy), env.view_angle, hang, env.view_distance,
                                  Theme.FOV_HELPER, scale, outline_rgba=Theme.FOV_OUT_HELPER)
            draw_sector_alpha(frame, tuple(env.red_pos), env.view_angle, env.red_angle, env.view_distance,
                              Theme.FOV_RED, scale, outline_rgba=Theme.FOV_OUT_RED)

        d2 = ImageDraw.Draw(frame, "RGBA")

        # путь A*
        if ctrl.path and ctrl.path_idx < len(ctrl.path):
            pts = ctrl.path[ctrl.path_idx:]
            if len(pts) > 1:
                line = [(y*scale + scale//2, x*scale + scale//2) for (x, y) in pts]
                d2.line(line, fill=Theme.PATH, width=max(2, scale//3))

        # цель
        if ctrl.target is not None:
            tx, ty = ctrl.target
            pad = max(1, scale//6)
            d2.rectangle([ty*scale+pad, tx*scale+pad, (ty+1)*scale-1-pad, (tx+1)*scale-1-pad],
                         outline=Theme.TARGET, width=max(2, scale//3))

        # LOS от красного к цели
        if ctrl.target is not None:
            A = tuple(env.red_pos); B = ctrl.target
            is_clear = los_clear(env.grid, A, B)
            los_color = Theme.GOOD_LOS if is_clear else Theme.BAD_LOS
            cx = A[1]*scale + scale//2; cy = A[0]*scale + scale//2
            tx = B[1]*scale + scale//2; ty = B[0]*scale + scale//2
            d2.line([(cx, cy), (tx, ty)], fill=(*los_color, 220), width=max(2, scale//3))
            los_samples += 1
            if is_clear: los_green_ratio += 1

        # helpers
        for hx, hy in env.helpers_pos:
            stroke_circle(frame, hx, hy, scale, r_in=max(2, scale//3 - 1), fill=(*Theme.RED_HELPER, 255))

        # blues + короткая стрелка
        for (bx, by), bang in zip(env.blues_pos, env.blues_angle):
            stroke_circle(frame, bx, by, scale, r_in=max(2, scale//3 - 1), fill=(*Theme.BLUE, 255))
            cx = by*scale + scale//2; cy = bx*scale + scale//2
            tx = cx + int(round(1.8*scale*math.cos(math.radians(bang))))
            ty = cy + int(round(1.8*scale*math.sin(math.radians(bang))))
            d2.line([(cx, cy), (tx, ty)], fill=(*Theme.BLUE, 220), width=max(2, scale//3))

        # red
        rx, ry = env.red_pos
        stroke_circle(frame, rx, ry, scale, r_in=max(3, scale//2 - 1), fill=(*Theme.RED, 255))
        draw_arrow(ImageDraw.Draw(frame, "RGBA"), (rx, ry), env.red_angle, scale, (*Theme.RED, 220),
                   length_cells=3.0, width=max(2, scale//3))

        # вклеиваем основную карту
        full.alpha_composite(frame, (0, 0))

        # ----- правая колонка -----
        side_x0 = main_w + margin

        def make_panel(title: str, obst_mask: np.ndarray, free_mask: np.ndarray,
                       new_mask: np.ndarray, fov_overlays: List[Tuple[Tuple[int,int], float, Tuple[int,int,int,int]]]):
            panel = Image.new("RGBA", (W*mini_scale, H*mini_scale), (248, 250, 252, 255))
            dr = ImageDraw.Draw(panel, "RGBA")
            # free
            for x, y in np.argwhere(free_mask == 1):
                col = Theme.EXP_FREE_RED if "Red" in title else Theme.EXP_FREE_BLUE
                dr.rectangle([y*mini_scale, x*mini_scale, (y+1)*mini_scale-1, (x+1)*mini_scale-1],
                             fill=col + (110,))
            # obstacles
            for x, y in np.argwhere(obst_mask == 1):
                dr.rectangle([y*mini_scale, x*mini_scale, (y+1)*mini_scale-1, (x+1)*mini_scale-1], fill=Theme.OBSTACLE)
            # new info contour
            for x, y in np.argwhere(new_mask == 1):
                dr.rectangle([y*mini_scale, x*mini_scale, (y+1)*mini_scale-1, (x+1)*mini_scale-1],
                             outline=(255, 140, 0, 230), width=1)
            # FOV overlays
            for (cx, cy), ang, col in fov_overlays:
                ov = Image.new("RGBA", panel.size, (0,0,0,0))
                draw_sector_alpha(ov, (cx, cy), env.view_angle, ang, env.view_distance, col, mini_scale)
                panel.alpha_composite(ov)
            # рамка + заголовок
            dr.rectangle([0,0, panel.width-1, panel.height-1], outline=(120,120,120,255), width=1)
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", 10 + mini_scale//3)
            except:
                font = ImageFont.load_default()
            bbox = dr.textbbox((0,0), title, font=font)
            dr.rectangle([4,4, 8+bbox[2], 6+bbox[3]], fill=(255,255,255,225))
            dr.text((6,5), title, fill=Theme.TEXT, font=font)
            return panel

        fov_red_list = [ (tuple(env.red_pos), env.red_angle, Theme.FOV_RED) ]
        for (hx, hy), hang in zip(env.helpers_pos, getattr(env, "helpers_angle", [])):
            fov_red_list.append(((hx, hy), hang, Theme.FOV_HELPER))
        fov_blue_list = [ ((bx,by), bang, Theme.FOV_BLUE) for (bx,by), bang in zip(env.blues_pos, env.blues_angle) ]

        red_panel = make_panel("Red Knowledge", red_known_obst, red_known_free, newly_known_red, fov_red_list)
        blue_panel= make_panel("Blue Knowledge", blue_known_obst, blue_known_free, newly_known_blue, fov_blue_list)

        shared_obst = np.logical_or(red_known_obst, blue_known_obst).astype(np.uint8)
        shared_free = np.logical_or(red_known_free, blue_known_free).astype(np.uint8)
        shared_panel= make_panel("Shared Knowledge", shared_obst, shared_free, np.zeros_like(shared_free), [])

        y_cursor = 0
        full.alpha_composite(red_panel, (side_x0, y_cursor)); y_cursor += red_panel.height + 8
        full.alpha_composite(blue_panel,(side_x0, y_cursor)); y_cursor += blue_panel.height + 8

        # авто-ужатие shared, если не помещается
        available = frame_h - y_cursor - 10
        if available < 40:
            available = 40
        if shared_panel.height > available:
            shared_panel = shared_panel.resize((side_w, available), Image.NEAREST)

        full.alpha_composite(shared_panel, (side_x0, y_cursor)); y_cursor += shared_panel.height + 10

        # HUD
        hud = Image.new("RGBA", (side_w, 130), (255,255,255,235))
        dh = ImageDraw.Draw(hud, "RGBA")
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 11 + mini_scale//3)
        except:
            font = ImageFont.load_default()

        if ctrl.target is not None:
            desired = angle_to(tuple(env.red_pos), ctrl.target)
            dtheta = min((desired - env.red_angle) % 360, (env.red_angle - desired) % 360)
        else:
            dtheta = float('nan')

        def red_sees_blue() -> int:
            cnt = 0
            for (bx,by) in env.blues_pos:
                if env._cell_in_fov(env.red_pos[0], env.red_pos[1], bx, by, env.red_angle):  # type: ignore
                    cnt += 1
            return cnt

        visible_now = red_sees_blue()
        if visible_now: seen_any_blue_this_ep += 1

        dh.text((8,6), f"dist to target: {cur_dist:.2f}" if ctrl.target else "dist to target: —", fill=Theme.TEXT, font=font)
        dh.text((8,24), f"Δdist/frame: {delta_d:+.3f}", fill=Theme.TEXT, font=font)
        dh.text((8,42), f"dθ to target: {dtheta:.1f}°" if ctrl.target else "dθ to target: —", fill=Theme.TEXT, font=font)
        dh.text((8,60), f"visible blues: {visible_now}", fill=Theme.TEXT, font=font)
        dh.text((8,78), f"replans: {ctrl.replans}", fill=Theme.TEXT, font=font)
        dh.text((8,96), f"timers: det={getattr(env,'detection_timer',-1)} | find={getattr(env,'finder_timer',-1)}", fill=Theme.TEXT, font=font)

        if ctrl.target:
            bar_w = 140; bar_h = 8; x0 = hud.width - bar_w - 12; y0 = 18
            dh.rectangle([x0, y0, x0+bar_w, y0+bar_h], outline=(120,120,120,255), width=1)
            maxd = max(H, W)
            fill_w = int(bar_w * (1.0 - max(0.0, min(1.0, cur_dist/maxd))))
            dh.rectangle([x0, y0, x0+fill_w, y0+bar_h], fill=(0,170,90,220))

        full.alpha_composite(hud, (side_x0, y_cursor)); y_cursor += hud.height + 8

        # timeline
        events_panel = Image.new("RGBA", (side_w, 110), (255,255,255,235))
        de = ImageDraw.Draw(events_panel, "RGBA")
        de.text((8,6), "Events:", fill=Theme.TEXT, font=font)
        for i, ev in enumerate(list(events)[:6]):
            de.text((16, 26 + i*16), f"• {ev}", fill=Theme.TEXT, font=font)
        full.alpha_composite(events_panel, (side_x0, y_cursor)); y_cursor += events_panel.height + 6

        # титул в правом верхнем углу правой колонки
        title = f"{scen_name} | step={env.steps} | blues={len(env.blues_pos)}"
        tb = d_full.textbbox((0,0), title, font=font)
        pad = 6
        tx = side_x0 + side_w - (tb[2]-tb[0]) - 2*pad - 8
        ty = 8
        d_full.rounded_rectangle([tx,ty, tx+(tb[2]-tb[0])+2*pad, ty+(tb[3]-tb[1])+2*pad],
                                 radius=6, fill=(255,255,255,235))
        d_full.text((tx+pad, ty+pad), title, fill=Theme.TEXT, font=font)

        # ===== ЛЕГЕНДА ВНИЗУ =====
        legend_lines = [
            ("Red (agent)", (*Theme.RED, 255)),
            ("Helpers", (*Theme.RED_HELPER, 255)),
            ("Blues (opponents)", (*Theme.BLUE, 255)),
            ("FOV Red", Theme.FOV_RED),
            ("FOV Helpers", Theme.FOV_HELPER),
            ("FOV Blue", Theme.FOV_BLUE),
            ("Visited (heat)", (*Theme.VISITED, 180)),
            ("Explored Free (Red)", (*Theme.EXP_FREE_RED, 180)),
            ("Explored Free (Blue)", (*Theme.EXP_FREE_BLUE, 180)),
            ("Obstacle", (*Theme.OBSTACLE, 255)),
            ("Path to target", (*Theme.PATH, 255)),
            ("LOS: green=clear / red=blocked", (0,0,0,0)),
        ]
        legend_img = build_legend_image(full.width, legend_lines, max(7, scale - 1))
        final_frame = compose_with_legend_bottom(full, legend_img, margin=6)

        frames.append(np.asarray(final_frame))

        # шаг симуляции
        action = ctrl.choose_action(env, events)
        obs, reward, terminated, truncated, info = env.step(action)
        if reward < -0.49:
            events.appendleft("Penalty/Collision")
        if terminated or truncated:
            events.appendleft("Episode end")
        t += 1

    # финальный бокс-итоги (на последнем кадре)
    if frames:
        fin = Image.fromarray(frames[-1]).convert("RGBA")
        dfin = ImageDraw.Draw(fin, "RGBA")
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 12)
        except:
            font = ImageFont.load_default()
        win = bool(info.get("is_win", False))
        loss= bool(info.get("is_loss", False))
        los_rate = (los_green_ratio/max(1, los_samples))*100.0
        explored_pct = 100.0 * ((red_known_free.sum()+red_known_obst.sum()) / (H*W*1.0))
        lines = [
            "EPISODE SUMMARY",
            f"Result: {'WIN' if win else 'LOSS' if loss else '—'}",
            f"Mean LOS clear: {los_rate:.1f}%",
            f"Seen blues (frames): {seen_any_blue_this_ep}",
            f"Replans: {ctrl.replans}",
            f"Explored (red) %cells: {explored_pct:.1f}%",
        ]
        w = max(dfin.textlength(s, font=font) for s in lines) + 24
        h = 10 + len(lines)*18 + 10
        x0 = fin.width - w - 10; y0 = fin.height - h - 10
        dfin.rounded_rectangle([x0,y0,x0+w,y0+h], radius=10, fill=(255,255,255,240), outline=(120,120,120,255), width=1)
        for i, s in enumerate(lines):
            dfin.text((x0+12, y0+12+i*18), s, fill=Theme.TEXT, font=font)
        frames.append(np.asarray(fin))

    out_path = os.path.join(out_dir, f"{scen_name}.gif")
    imageio.mimsave(out_path, frames, duration=max(0.02, 1.0/max(1, fps)))
    with open(os.path.join(out_dir, f"{scen_name}.json"), "w") as f:
        json.dump(dict(scenario=scen_name, kwargs=env_kwargs, steps=t,
                       terminated=bool(terminated), truncated=bool(truncated), final_info=info), f, indent=2)
    print(f"[OK] {scen_name} -> {out_path}")
    return out_path



# ============================ Scenarios ===========================

def scenario_bank() -> Dict[str, Dict[str, Any]]:
    return {
        "baseline": dict(grid_size=64, view_distance=8, view_angle=90, max_steps=2000,
                         num_blue=1, blue_flee_on_seen=True, patrol_noise=30.0,
                         dynamic_obstacles_steps=0, obstacles_density=0.2),
        "reduced_fov": dict(grid_size=64, view_distance=5, view_angle=60, max_steps=2000,
                            num_blue=1, blue_flee_on_seen=True, patrol_noise=15.0,
                            dynamic_obstacles_steps=0, obstacles_density=0.2),
        "dynamic_obstacles": dict(grid_size=64, view_distance=8, view_angle=90, max_steps=2000,
                                  num_blue=1, blue_flee_on_seen=True, patrol_noise=30.0,
                                  dynamic_obstacles_steps=50, obstacles_density=0.25),
        "multiple_blue_3": dict(grid_size=64, view_distance=8, view_angle=90, max_steps=2000,
                                num_blue=3, blue_flee_on_seen=True, patrol_noise=30.0,
                                dynamic_obstacles_steps=0, obstacles_density=0.2),
        "labyrinth_dense": dict(grid_size=64, view_distance=8, view_angle=90, max_steps=2000,
                                num_blue=1, blue_flee_on_seen=True, patrol_noise=15.0,
                                dynamic_obstacles_steps=0, obstacles_density=0.65),
        "abl_no_helpers": dict(grid_size=64, view_distance=8, view_angle=90, max_steps=2000,
                               num_blue=1, blue_flee_on_seen=True, patrol_noise=15.0,
                               dynamic_obstacles_steps=0, obstacles_density=0.3,
                               use_helpers=False, num_helpers=0),
        "abl_no_last_seen": dict(grid_size=64, view_distance=8, view_angle=90, max_steps=2000,
                                 num_blue=1, blue_flee_on_seen=True, patrol_noise=15.0,
                                 dynamic_obstacles_steps=0, obstacles_density=0.3,
                                 use_last_seen=False),
        "abl_no_memory": dict(grid_size=64, view_distance=8, view_angle=90, max_steps=2000,
                              num_blue=1, blue_flee_on_seen=True, patrol_noise=15.0,
                              dynamic_obstacles_steps=0, obstacles_density=0.3,
                              use_obstacle_memory=False),
        "big_map": dict(grid_size=80, view_distance=10, view_angle=90, max_steps=2400,
                        num_blue=2, blue_flee_on_seen=True, patrol_noise=15.0,
                        dynamic_obstacles_steps=0, obstacles_density=0.25),
        "hi_noise_patrol": dict(grid_size=64, view_distance=8, view_angle=90, max_steps=2000,
                                num_blue=2, blue_flee_on_seen=True, patrol_noise=45.0,
                                dynamic_obstacles_steps=0, obstacles_density=0.25),
        "no_flee": dict(grid_size=64, view_distance=8, view_angle=90, max_steps=2000,
                        num_blue=2, blue_flee_on_seen=False, patrol_noise=15.0,
                        dynamic_obstacles_steps=0, obstacles_density=0.25),
    }


# ============================== CLI ===============================

def main():
    parser = argparse.ArgumentParser("Beautiful GIFs: FOV outlines, LOS, knowledge mini-maps, HUD, timeline")
    parser.add_argument("--out", type=str, default="ideal_gifs", help="Output dir")
    parser.add_argument("--steps", type=int, default=800, help="Max steps per GIF")
    parser.add_argument("--seed", type=int, default=12345, help="Env seed")
    parser.add_argument("--scale", type=int, default=10, help="Pixels per cell for main map")
    parser.add_argument("--mini-scale", type=int, default=5, help="Pixels per cell for mini-maps")
    parser.add_argument("--fps", type=int, default=18, help="FPS")
    parser.add_argument("--grid", type=int, default=8, help="Grid spacing (0=off)")
    parser.add_argument("--no-fov", action="store_true", help="Disable FOV rendering")
    parser.add_argument("--only", nargs="*", default=None, help="Subset of scenario names")
    args = parser.parse_args()

    bank = scenario_bank()
    names = args.only if args.only else list(bank.keys())

    for name in names:
        env_kwargs = dict(bank[name])
        subdir = name.split("_")[0]
        out_dir = os.path.join(args.out, subdir)
        simulate_ideal_gif(
            scen_name=name,
            env_kwargs=env_kwargs,
            out_dir=out_dir,
            steps_limit=args.steps,
            seed=args.seed,
            scale=args.scale,
            mini_scale=args.mini_scale,
            fps=args.fps,
            show_grid_every=args.grid,
            draw_fovs=not args.no_fov,
        )

if __name__ == "__main__":
    main()
