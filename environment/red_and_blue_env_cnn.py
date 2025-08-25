import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any


class RedAndBlueEnv(gym.Env):
    """
    CNN-совместимая среда.
    Поддерживает:
      - Параметры обзора, плотность препятствий, динамические препятствия
      - Несколько синих
      - Режим «убегать при обнаружении»
      - Абляции механик (helpers / last_seen / obstacle_memory / detection_timer)
      - Веса компонентов награды (для абляций reward shaping)
      - Быстрый рендер в RGB массив (get_frame) — подходит для GIF

    Обсервация: (6, H, W), uint8
      0: локальная карта препятствий (то, что видели)          [0/255]
      1: посещённые клетки                                      [0/255]
      2: "last_seen" врага                                      [0/255]
      3: one-hot позиция красного                               [0/255]
      4: sin(angle_red)                                         [0..255]
      5: cos(angle_red)                                         [0..255]
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        grid_size: int = 64,
        view_distance: int = 8,
        view_angle: float = 90,
        max_steps: int = 2000,
        seed: int | None = None,
        # поведение противника
        num_blue: int = 1,
        blue_flee_on_seen: bool = True,
        patrol_noise: float = 30.0,
        dynamic_obstacles_steps: int = 0,
        obstacles_density: float = 0.2,
        # абляции механик
        use_helpers: bool = True,
        num_helpers: int = 2,
        use_last_seen: bool = True,
        use_obstacle_memory: bool = True,
        use_detection_timer: bool = True,
        # веса награды (для абляций reward shaping)
        reward_weights: Dict[str, float] | None = None,
    ):
        super().__init__()
        self.grid_size = int(grid_size)
        self.view_distance = int(view_distance)
        self.view_angle = float(view_angle)
        self.max_steps = int(max_steps)
        self.num_blue = max(1, int(num_blue))
        self.blue_flee_on_seen = bool(blue_flee_on_seen)
        self.patrol_noise = float(patrol_noise)
        self.dynamic_obstacles_steps = int(dynamic_obstacles_steps)
        self.obstacles_density = float(np.clip(obstacles_density, 0.0, 1.0))

        # абляции
        self.use_helpers = bool(use_helpers)
        self.num_helpers = max(0, int(num_helpers)) if self.use_helpers else 0
        self.use_last_seen = bool(use_last_seen)
        self.use_obstacle_memory = bool(use_obstacle_memory)
        self.use_detection_timer = bool(use_detection_timer)

        # веса награды
        default_rw = dict(
            discover_bonus=0.0,     # бонус при первом обнаружении
            proximity=1.0,          # ближе к ближайшему синему
            looking_bonus=0.5,      # смотришь на цель
            collision_penalty=-0.5, # штраф за шаг в препятствие
            step_penalty=-0.01,     # штраф за шаг в «преследовании»
            explore_new=0.5,        # посетил новую клетку
            revisit_penalty=-0.01,  # топчешься
            explore_step_penalty=-0.005,  # базовый штраф в режиме исследования
        )
        if reward_weights:
            default_rw.update(reward_weights)
        self.rw = default_rw

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(6, self.grid_size, self.grid_size), dtype=np.uint8
        )

        self._rng = np.random.RandomState(seed)
        self._reset_internal()

    # ------------- Gym API -------------
    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        self._reset_internal()
        return self._get_obs(), {}

    def step(self, action):
        self._apply_red_action(action)
        if self.use_helpers:
            self._move_all_helpers()
        self._move_blues()

        if self.dynamic_obstacles_steps > 0 and (self.steps % self.dynamic_obstacles_steps == 0):
            self._jiggle_obstacles()

        self._update_maps_and_discovery()

        reward = self._reward()
        self.steps += 1

        is_win = self._red_sees_any_blue()
        is_loss = self._any_blue_sees_red()
        terminated = bool(is_win or is_loss)

        truncated = False
        if self.use_detection_timer and any(self.blue_discovered_list):
            self.detection_timer -= 1
            if self.detection_timer <= 0:
                truncated = True
                reward += 0.0  # сам штраф заложен в shaping, можно не добавлять
        self.finder_timer -= 1
        if self.finder_timer <= 0:
            truncated = True

        if terminated:
            reward += 100.0 if is_win else -100.0

        info = {
            "is_win": bool(is_win),
            "is_loss": bool(is_loss),
            "episode_length": self.steps,
        }
        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self):
        print(f"step={self.steps} red={self.red_pos} blues={self.blues_pos[:2]}...")

    # ------------- GIF helper -------------
    def get_frame(self):
        H = W = self.grid_size
        img = np.zeros((H, W, 3), dtype=np.uint8)
        img[:, :, :] = 210
        # препятствия — чёрные
        img[self.grid.astype(bool)] = (0, 0, 0)
        # last_seen — голубой
        if self.use_last_seen:
            img[self.last_seen_blue.astype(bool)] = (120, 180, 255)
        # visited — розовый
        img[self.visited_red.astype(bool)] = (255, 200, 220)
        # helpers — тёмно-красные
        for hx, hy in self.helpers_pos:
            img[hx, hy] = (180, 0, 0)
        # красный
        x, y = self.red_pos
        img[x, y] = (255, 0, 0)
        # синие
        for bx, by in self.blues_pos:
            img[bx, by] = (0, 0, 255)
        return img

    # ------------- Internal -------------
    def _reset_internal(self):
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        self._place_obstacles()

        self.red_pos = self._random_empty_position()
        self.red_angle = self._rng.uniform(0, 360)

        self.blues_pos = [self._random_empty_position() for _ in range(self.num_blue)]
        self.blues_angle = [self._rng.uniform(0, 360) for _ in range(self.num_blue)]
        self.blue_spooked = [False for _ in range(self.num_blue)]
        self.blue_discovered_list = [False for _ in range(self.num_blue)]

        # helpers
        self.helpers_pos = []
        self.helpers_angle = []
        for _ in range(self.num_helpers):
            self.helpers_pos.append(self._random_empty_position())
            self.helpers_angle.append(self._rng.uniform(0, 360))

        self.local_map_red = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        self.visited_red = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.last_seen_blue = np.zeros((self.grid_size, self.grid_size), dtype=bool)

        self.detection_timer = 200
        self.finder_timer = self.max_steps
        self.steps = 0

    def _apply_red_action(self, action):
        if action == 0:
            self.red_pos[0] = max(0, self.red_pos[0] - 1)
        elif action == 1:
            self.red_pos[0] = min(self.grid_size - 1, self.red_pos[0] + 1)
        elif action == 2:
            self.red_pos[1] = max(0, self.red_pos[1] - 1)
        elif action == 3:
            self.red_pos[1] = min(self.grid_size - 1, self.red_pos[1] + 1)
        elif action == 4:
            self.red_angle = (self.red_angle + 45) % 360

    def _move_all_helpers(self):
        for i in range(len(self.helpers_pos)):
            self.helpers_angle[i] = self._move_helper(self.helpers_pos[i], self.helpers_angle[i])

    def _move_blues(self):
        for i in range(self.num_blue):
            pos = self.blues_pos[i]
            ang = self.blues_angle[i]
            if self.blue_flee_on_seen and (self.blue_discovered_list[i] or self.blue_spooked[i]):
                dx, dy = np.array(pos) - np.array(self.red_pos)
                flee = (np.degrees(np.arctan2(dy, dx)) + 360) % 360
                ang = flee + self._rng.uniform(-self.patrol_noise, self.patrol_noise)
            else:
                # патруль
                if self._rng.rand() < 0.6:
                    ang = (ang + self._rng.choice([-45, 0, 45])) % 360
                ang = (ang + self._rng.uniform(-self.patrol_noise, self.patrol_noise)) % 360
            step = self._angle_to_step(ang)
            nx = int(np.clip(pos[0] + step[0], 0, self.grid_size - 1))
            ny = int(np.clip(pos[1] + step[1], 0, self.grid_size - 1))
            if self.grid[nx, ny] == 0:
                self.blues_pos[i] = [nx, ny]
            self.blues_angle[i] = ang

    def _move_helper(self, pos, ang):
        action = self._rng.choice(["move", "turn_left", "turn_right"])
        if action == "move":
            step = self._angle_to_step(ang)
            nx = int(np.clip(pos[0] + step[0], 0, self.grid_size - 1))
            ny = int(np.clip(pos[1] + step[1], 0, self.grid_size - 1))
            if self.grid[nx, ny] == 0:
                pos[0], pos[1] = nx, ny
        elif action == "turn_left":
            ang = (ang - 45) % 360
        else:
            ang = (ang + 45) % 360
        return ang

    def _angle_to_step(self, angle_deg):
        dx = int(np.round(np.cos(np.radians(angle_deg))))
        dy = int(np.round(np.sin(np.radians(angle_deg))))
        return np.array([dx, dy], dtype=int)

    def _update_maps_and_discovery(self):
        self.visited_red[self.red_pos[0], self.red_pos[1]] = True

        # кто смотрит
        viewers = [(self.red_pos, self.red_angle)]
        if self.use_helpers:
            viewers += list(zip(self.helpers_pos, self.helpers_angle))

        # отметки видимости и "испуганности"
        for b_idx, (bx, by) in enumerate(self.blues_pos):
            seen = False
            for (x, y), ang in viewers:
                if self._cell_in_fov(x, y, bx, by, ang):
                    if self.use_last_seen:
                        self.last_seen_blue[bx, by] = True
                    seen = True
            if seen:
                self.blue_discovered_list[b_idx] = True
                self.blue_spooked[b_idx] = True

        # запоминание препятствий — только если не отключено абляцией
        if self.use_obstacle_memory:
            for (x, y), ang in viewers:
                self._stamp_fov_obstacles(self.local_map_red, (x, y), ang)

    def _stamp_fov_obstacles(self, local_map, pos, angle):
        x, y = pos
        R = self.view_distance
        for i in range(-R, R + 1):
            for j in range(-R, R + 1):
                nx, ny = x + i, y + j
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    if self._cell_in_fov(x, y, nx, ny, angle) and self.grid[nx, ny] == 1:
                        local_map[nx, ny] = 1

    def _cell_in_fov(self, x, y, nx, ny, angle):
        dx, dy = nx - x, ny - y
        dist2 = dx * dx + dy * dy
        if dist2 > self.view_distance * self.view_distance:
            return False
        if dx == 0 and dy == 0:
            return True
        ang_to = (np.degrees(np.arctan2(dy, dx)) + 360) % 360
        diff = abs((angle - ang_to + 360) % 360)
        return diff <= self.view_angle / 2

    def _red_sees_any_blue(self):
        for (bx, by) in self.blues_pos:
            if self._cell_in_fov(self.red_pos[0], self.red_pos[1], bx, by, self.red_angle):
                return True
        return False

    def _any_blue_sees_red(self):
        for (bx, by), bang in zip(self.blues_pos, self.blues_angle):
            if self._cell_in_fov(bx, by, self.red_pos[0], self.red_pos[1], bang):
                return True
        return False

    def _place_obstacles(self):
        base = int(np.interp(self.obstacles_density, [0, 1], [6, 28]))
        for _ in range(base):
            x, y = self._rng.randint(0, self.grid_size, size=2)
            w, h = self._rng.randint(1, max(2, self.grid_size // 12), size=2)
            self.grid[x:x + w, y:y + h] = 1

    def _jiggle_obstacles(self):
        for _ in range(10):
            x, y = self._rng.randint(0, self.grid_size, size=2)
            if self.grid[x, y] == 1 and self._rng.rand() < 0.5:
                self.grid[x, y] = 0
            elif self.grid[x, y] == 0 and self._rng.rand() < 0.2:
                self.grid[x, y] = 1

    def _random_empty_position(self):
        for _ in range(10000):
            pos = [self._rng.randint(0, self.grid_size), self._rng.randint(0, self.grid_size)]
            if self.grid[pos[0], pos[1]] == 0:
                return pos
        return [0, 0]

    def _encode_angle(self, angle_deg):
        s = (np.sin(np.radians(angle_deg)) + 1.0) * 0.5
        c = (np.cos(np.radians(angle_deg)) + 1.0) * 0.5
        return s, c

    def _get_obs(self):
        H = W = self.grid_size
        obs = np.zeros((6, H, W), dtype=np.uint8)
        obs[0] = self.local_map_red * 255 if self.use_obstacle_memory else 0
        obs[1] = self.visited_red.astype(np.uint8) * 255
        obs[2] = self.last_seen_blue.astype(np.uint8) * 255 if self.use_last_seen else 0

        red_pos_ch = np.zeros((H, W), dtype=np.uint8)
        red_pos_ch[self.red_pos[0], self.red_pos[1]] = 255
        obs[3] = red_pos_ch

        s, c = self._encode_angle(self.red_angle)
        obs[4] = np.full((H, W), int(s * 255), dtype=np.uint8)
        obs[5] = np.full((H, W), int(c * 255), dtype=np.uint8)
        return obs

    def _reward(self):
        r = 0.0
        dists = [np.linalg.norm(np.array(bp) - np.array(self.red_pos)) for bp in self.blues_pos]
        nearest = float(min(dists)) if dists else float(self.grid_size)

        # бонус за первое обнаружение
        if self.rw["discover_bonus"] > 0 and any(self.blue_discovered_list) and self.steps == 1:
            r += self.rw["discover_bonus"]

        if any(self.blue_discovered_list):
            r += self.rw["proximity"] * max(0.0, 1.0 - nearest / self.grid_size)

            bx, by = self.blues_pos[int(np.argmin(dists))]
            dx, dy = np.array([bx, by]) - np.array(self.red_pos)
            ang_to = (np.degrees(np.arctan2(dy, dx)) + 360) % 360
            diff = abs((self.red_angle - ang_to + 360) % 360)
            if diff <= self.view_angle / 2:
                r += self.rw["looking_bonus"]

            if self.grid[self.red_pos[0], self.red_pos[1]] == 1:
                r += self.rw["collision_penalty"]

            r += self.rw["step_penalty"]
        else:
            # исследование
            if not self.visited_red[self.red_pos[0], self.red_pos[1]]:
                r += self.rw["explore_new"]
            else:
                r += self.rw["revisit_penalty"]
            r += self.rw["explore_step_penalty"]
        return r
