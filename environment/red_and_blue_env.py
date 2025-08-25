import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib
matplotlib.use('TkAgg')  # Используем TkAgg для интерактивных графиков
import matplotlib.pyplot as plt

plt.ion()  # Включение интерактивного режима

class RedAndBlueEnv(gym.Env):
    def __init__(self):
        super(RedAndBlueEnv, self).__init__()

        self.grid_size = 100
        self.view_distance = 10
        self.view_angle = 90
        
        self.action_space = spaces.Discrete(5)  # 0: вверх, 1: вниз, 2: влево, 3: вправо, 4: поворот
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.grid_size * self.grid_size * 3 * 2,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed)  # Устанавливаем seed для генератора случайных чисел
        self.grid = np.zeros((self.grid_size, self.grid_size))
        self.red_pos = self._random_empty_position()
        self.blue_pos = self._random_empty_position()

        self.red_angle = np.random.uniform(0, 360)
        self.blue_angle = np.random.uniform(0, 360)

        self.helper1_red_pos = self._random_empty_position()
        self.helper2_red_pos = self._random_empty_position()

        self.helper1_blue_pos = self._random_empty_position()
        self.helper2_blue_pos = self._random_empty_position()

        self.helper1_red_angle = np.random.uniform(0, 360)
        self.helper2_red_angle = np.random.uniform(0, 360)

        self.helper1_blue_angle = np.random.uniform(0, 360)
        self.helper2_blue_angle = np.random.uniform(0, 360)

        self._place_obstacles()

        # Локальные карты для игроков
        self.local_map_red = np.zeros((self.grid_size, self.grid_size))
        self.local_map_blue = np.zeros((self.grid_size, self.grid_size))
        self.visited_red = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.visited_blue = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.last_seen_blue = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.last_seen_red = np.zeros((self.grid_size, self.grid_size), dtype=bool)

        self.blue_discovered = False
        self.red_discovered = False

        return self._get_obs(), {}  # Возвращаем наблюдения и пустую информацию


    def step(self, action):
        if action == 0:  # вверх
            self.red_pos[0] = max(0, self.red_pos[0] - 1)
        elif action == 1:  # вниз
            self.red_pos[0] = min(self.grid_size - 1, self.red_pos[0] + 1)
        elif action == 2:  # влево
            self.red_pos[1] = max(0, self.red_pos[1] - 1)
        elif action == 3:  # вправо
            self.red_pos[1] = min(self.grid_size - 1, self.red_pos[1] + 1)
        elif action == 4:  # Поворот
            self.red_angle = (self.red_angle + 45) % 360

        # Помощники двигаются случайным образом
        self.helper1_red_angle = self._move_helper(self.helper1_red_pos, self.helper1_red_angle)
        self.helper2_red_angle = self._move_helper(self.helper2_red_pos, self.helper2_red_angle)

        self.helper1_blue_angle = self._move_helper(self.helper1_blue_pos, self.helper1_blue_angle)
        self.helper2_blue_angle = self._move_helper(self.helper2_blue_pos, self.helper2_blue_angle)

        # Обновление карт для игроков
        red_area = self._get_red_area(self.red_pos, self.red_angle)
        blue_area = self._get_red_area(self.blue_pos, self.blue_angle)

        helper1_red_area = self._get_helper_area(self.helper1_red_pos, self.helper1_red_angle)
        helper2_red_area = self._get_helper_area(self.helper2_red_pos, self.helper2_red_angle)

        helper1_blue_area = self._get_helper_area(self.helper1_blue_pos, self.helper1_blue_angle)
        helper2_blue_area = self._get_helper_area(self.helper2_blue_pos, self.helper2_blue_angle)

        self._update_local_map(self.local_map_red, self.visited_red, self.last_seen_blue, helper1_red_area, helper2_red_area, red_area, self.blue_pos)
        self._update_local_map(self.local_map_blue, self.visited_blue, self.last_seen_red, helper1_blue_area, helper2_blue_area, blue_area, self.red_pos)

        done = self._check_termination()
        reward = self.reward_function()
        
        if done:
            reward = 10

        return self._get_obs(), reward, done, False, {}

    def _update_local_map(self, local_map, visited, last_seen, area1, area2, main_area, enemy_pos):
        # Обновляем локальную карту на основе данных от игрока и помощников
        # 1. Отметка препятствий и посещенных клеток
        for area, pos_angle in zip(
            [main_area, area1, area2],
            [(self.red_pos, self.red_angle) if local_map is self.local_map_red else (self.blue_pos, self.blue_angle),
            (self.helper1_red_pos, self.helper1_red_angle) if local_map is self.local_map_red else (self.helper1_blue_pos, self.helper1_blue_angle),
            (self.helper2_red_pos, self.helper2_red_angle) if local_map is self.local_map_red else (self.helper2_blue_pos, self.helper2_blue_angle)]
        ):
            (x, y), angle = pos_angle  # Распаковываем координаты и угол
            for i in range(-self.view_distance, self.view_distance + 1):
                for j in range(-self.view_distance, self.view_distance + 1):
                    nx, ny = x + i, y + j
                    if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                        if area[i + self.view_distance, j + self.view_distance] == 1:  # Обнаружено препятствие
                            local_map[nx, ny] = 1
                        visited[nx, ny] = True

        # 2. Отметка последнего видимого местоположения противника
        for (x, y), angle in [
            (self.red_pos, self.red_angle) if local_map is self.local_map_red else (self.blue_pos, self.blue_angle),
            (self.helper1_red_pos, self.helper1_red_angle) if local_map is self.local_map_red else (self.helper1_blue_pos, self.helper1_blue_angle),
            (self.helper2_red_pos, self.helper2_red_angle) if local_map is self.local_map_red else (self.helper2_blue_pos, self.helper2_blue_angle)
        ]:
            if self._is_in_view((x, y), angle, enemy_pos):
                last_seen[enemy_pos[0], enemy_pos[1]] = True



    def _get_obs(self):
        obs_red = np.stack([self.local_map_red, self.visited_red.astype(float), self.last_seen_blue.astype(float)], axis=-1)
        obs_blue = np.stack([self.local_map_blue, self.visited_blue.astype(float), self.last_seen_red.astype(float)], axis=-1)
        # Объединяем наблюдения
        combined_obs = np.concatenate([obs_red.flatten(), obs_blue.flatten()])
        return combined_obs

    def render(self, mode='human'):
        plt.clf()
        # Отображение основной карты
        plt.subplot(1, 3, 1)
        plt.imshow(self.grid, cmap="gray")

        # Отображаем красного игрока и его помощников
        plt.scatter(self.red_pos[1], self.red_pos[0], color="red", s=50, label="Red Player", zorder=5)
        plt.scatter(self.helper1_red_pos[1], self.helper1_red_pos[0], color="darkred", s=30, label="Red Helper 1", zorder=5)
        plt.scatter(self.helper2_red_pos[1], self.helper2_red_pos[0], color="darkred", s=30, label="Red Helper 2", zorder=5)
        self._draw_view(self.red_pos, self.red_angle, "red")
        self._draw_view(self.helper1_red_pos, self.helper1_red_angle, "darkred")
        self._draw_view(self.helper2_red_pos, self.helper2_red_angle, "darkred")

        # Отображаем синего игрока и его помощников
        plt.scatter(self.blue_pos[1], self.blue_pos[0], color="blue", s=50, label="Blue Player", zorder=5)
        plt.scatter(self.helper1_blue_pos[1], self.helper1_blue_pos[0], color="darkblue", s=30, label="Blue Helper 1", zorder=5)
        plt.scatter(self.helper2_blue_pos[1], self.helper2_blue_pos[0], color="darkblue", s=30, label="Blue Helper 2", zorder=5)
        self._draw_view(self.blue_pos, self.blue_angle, "blue")
        self._draw_view(self.helper1_blue_pos, self.helper1_blue_angle, "darkblue")
        self._draw_view(self.helper2_blue_pos, self.helper2_blue_angle, "darkblue")

        plt.title("Main Map")
        plt.legend()

        # Локальная карта красного
        plt.subplot(1, 3, 2)
        plt.imshow(self.local_map_red, cmap="Reds", alpha=0.5)
        plt.imshow(self.last_seen_blue, cmap="Blues", alpha=0.3)
        plt.scatter(self.helper1_red_pos[1], self.helper1_red_pos[0], color="darkred", s=30, label="Helper 1", zorder=5)
        plt.scatter(self.helper2_red_pos[1], self.helper2_red_pos[0], color="darkred", s=30, label="Helper 2", zorder=5)
        self._draw_view(self.helper1_red_pos, self.helper1_red_angle, "darkred")
        self._draw_view(self.helper2_red_pos, self.helper2_red_angle, "darkred")
        plt.title("Red Local Map")

        # Локальная карта синего
        plt.subplot(1, 3, 3)
        plt.imshow(self.local_map_blue, cmap="Blues", alpha=0.5)
        plt.imshow(self.last_seen_red, cmap="Reds", alpha=0.3)
        plt.scatter(self.helper1_blue_pos[1], self.helper1_blue_pos[0], color="darkblue", s=30, label="Helper 1", zorder=5)
        plt.scatter(self.helper2_blue_pos[1], self.helper2_blue_pos[0], color="darkblue", s=30, label="Helper 2", zorder=5)
        self._draw_view(self.helper1_blue_pos, self.helper1_blue_angle, "darkblue")
        self._draw_view(self.helper2_blue_pos, self.helper2_blue_angle, "darkblue")
        plt.title("Blue Local Map")

        plt.draw()
        plt.pause(0.001)

    def _draw_view(self, pos, angle, color):
        x, y = pos
        for i in range(-self.view_distance, self.view_distance + 1):
            for j in range(-self.view_distance, self.view_distance + 1):
                if 0 <= x + i < self.grid_size and 0 <= y + j < self.grid_size:
                    dx, dy = x + i - x, y + j - y
                    dist = np.sqrt(dx ** 2 + dy ** 2)
                    if dist <= self.view_distance:
                        angle_to = (np.degrees(np.arctan2(dy, dx)) + 360) % 360
                        angle_diff = abs((angle - angle_to + 360) % 360)
                        if angle_diff <= self.view_angle / 2:
                            plt.plot(y + j, x + i, marker='o', color=color, alpha=0.2)

    def _place_obstacles(self):
        num_obstacles = np.random.randint(10, 30)
        for _ in range(num_obstacles):
            x, y = np.random.randint(0, self.grid_size, size=2)
            w, h = np.random.randint(1, 5, size=2)
            self.grid[x:x + w, y:y + h] = 1

    def _random_empty_position(self):
        while True:
            pos = [np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size)]
            if self.grid[pos[0], pos[1]] == 0:
                return pos

    def _move_helper(self, helper_pos, helper_angle):
        # Выбор действия: двигаться вперед, поворачивать влево или вправо
        action = np.random.choice(["move", "turn_left", "turn_right"])

        if action == "move":
            # Рассчитываем смещение по углу
            dx = int(round(np.cos(np.radians(helper_angle))))
            dy = int(round(np.sin(np.radians(helper_angle))))
            new_x = max(0, min(self.grid_size - 1, helper_pos[0] + dx))
            new_y = max(0, min(self.grid_size - 1, helper_pos[1] + dy))
            # Проверяем, что нет препятствия
            if self.grid[new_x, new_y] == 0:
                helper_pos[0], helper_pos[1] = new_x, new_y

        elif action == "turn_left":
            helper_angle = (helper_angle - 45) % 360  # Поворот против часовой стрелки
        elif action == "turn_right":
            helper_angle = (helper_angle + 45) % 360  # Поворот по часовой стрелке

        return helper_angle

    def _get_helper_area(self, pos, angle):
        # Функция для получения области видимости помощника (в пределах distance и angle)
        x, y = pos
        helper_area = np.zeros((2 * self.view_distance + 1, 2 * self.view_distance + 1))  # Задаем область видимости
        for i in range(-self.view_distance, self.view_distance + 1):
            for j in range(-self.view_distance, self.view_distance + 1):
                if 0 <= x + i < self.grid_size and 0 <= y + j < self.grid_size:
                    dx, dy = x + i - x, y + j - y
                    angle_to = (np.degrees(np.arctan2(dy, dx)) + 360) % 360
                    angle_diff = abs((angle - angle_to + 360) % 360)
                    if angle_diff <= self.view_angle / 2:
                        helper_area[i + self.view_distance, j + self.view_distance] = self.grid[x + i, y + j]
        return helper_area

    def _get_red_area(self, pos, angle):
        # Функция для получения области видимости красного игрока (в пределах distance и angle)
        x, y = pos
        red_area = np.zeros((2 * self.view_distance + 1, 2 * self.view_distance + 1))  # Задаем область видимости
        for i in range(-self.view_distance, self.view_distance + 1):
            for j in range(-self.view_distance, self.view_distance + 1):
                if 0 <= x + i < self.grid_size and 0 <= y + j < self.grid_size:
                    dx, dy = x + i - x, y + j - y
                    angle_to = (np.degrees(np.arctan2(dy, dx)) + 360) % 360
                    angle_diff = abs((angle - angle_to + 360) % 360)
                    if angle_diff <= self.view_angle / 2:
                        red_area[i + self.view_distance, j + self.view_distance] = self.grid[x + i, y + j]
        return red_area

    def reward_function(self):
        reward = 0

        if self.blue_discovered:
            # 1. Стремление двигаться к синему
            dx, dy = np.array(self.blue_pos) - np.array(self.red_pos)
            distance_to_blue = np.sqrt(dx ** 2 + dy ** 2)
            reward += -distance_to_blue / self.grid_size  # Чем ближе, тем больше награда

            # 2. Стимулирование взгляда на синего
            angle_to_blue = (np.degrees(np.arctan2(dy, dx)) + 360) % 360
            angle_diff = abs((self.red_angle - angle_to_blue + 360) % 360)
            if angle_diff <= self.view_angle / 2:
                reward += 1  # Дополнительное вознаграждение, если красный смотрит на синего

            # 3. Штрафы за столкновение с препятствиями
            if self.grid[self.red_pos[0], self.red_pos[1]] == 1:
                reward -= 1  # Штраф за столкновение с препятствием

            # 4. Малое вознаграждение за движение (не бездействие)
            reward -= 0.05  # Малое штрафное вознаграждение за каждое движение

        else:
            # Если синий не обнаружен, стимулируем исследование
            x, y = self.red_pos
            if not self.visited_red[x, y]:
                reward += 1  # Вознаграждение за посещение новой клетки
                self.visited_red[x, y] = True
            else:
                reward -= 0.01  # Небольшой штраф за повторное посещение

            # Вознаграждаем движение, если оно приближает к предполагаемой позиции синего
            dx, dy = np.array(self.blue_pos) - np.array(self.red_pos)
            distance_to_blue = np.sqrt(dx ** 2 + dy ** 2)
            reward += max(-distance_to_blue / (2 * self.grid_size), -0.5)  # Стимул двигаться к синему

        return reward


    def _check_termination(self):
        if self._is_in_view(self.red_pos, self.red_angle, self.blue_pos):
            return True  # Красный убил синего
        if self._is_in_view(self.blue_pos, self.blue_angle, self.red_pos):
            return True  # Синий убил красного
        return False

    def _is_in_view(self, pos1, angle1, pos2):
        # Вычисляем разницу в позициях
        dx, dy = np.array(pos2) - np.array(pos1)

        # Вычисляем расстояние между двумя точками
        distance = np.sqrt(dx ** 2 + dy ** 2)

        # Проверяем, находится ли цель в пределах дистанции обзора
        if distance > self.view_distance:
            return False  # Цель слишком далеко

        # Вычисляем угол между двумя точками
        angle_to = (np.degrees(np.arctan2(dy, dx)) + 360) % 360

        # Вычисляем разницу углов
        angle_diff = abs((angle1 - angle_to + 360) % 360)

        # Проверяем, попадает ли цель в угол обзора
        return angle_diff <= self.view_angle / 2


