import os
import io
import csv
import json
import time
import zipfile
import shutil
import random
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple

import numpy as np
import imageio.v2 as imageio
import requests

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from callbacks import VecEpisodeLogger
from environment.red_and_blue_env_cnn import RedAndBlueEnv

from make_ideal_gifs import simulate_ideal_gif


# ======= Глобальные настройки =======
RANDOM_SEED = 2025
MAX_EXPERIMENTS = 100

# базовые PPO параметры
BASE_N_ENVS = 8
BASE_N_STEPS = 128
BASE_N_EPOCHS = 6
BASE_DEVICE = "cuda"  # или "cpu"

# телеграм
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT  = os.getenv("TELEGRAM_CHAT_ID", "")

OUT_ROOT = "exp_runs"
GIF_STEPS = 600


# ========== Утилиты ==========
def ensure_clean_dir(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

def make_env_fn(seed_base: int, idx: int, env_kwargs: Dict[str, Any]):
    def _init():
        kw = dict(env_kwargs)
        kw["seed"] = seed_base + idx
        return RedAndBlueEnv(**kw)
    return _init

def send_file_telegram(path: str, caption: str = "") -> bool:
    if not TG_TOKEN or not TG_CHAT:
        print(f"[TG] skip send: {os.path.basename(path)} (no token/chat)")
        return False
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendDocument"
    try:
        with open(path, "rb") as f:
            files = {"document": (os.path.basename(path), f)}
            data = {"chat_id": TG_CHAT, "caption": caption}
            r = requests.post(url, files=files, data=data, timeout=180)
        print(f"[TG] {os.path.basename(path)} -> {r.status_code}")
        return r.status_code == 200
    except Exception as e:
        print(f"[TG] send error: {e}")
        return False

def record_gif(model: PPO, env_kwargs: Dict[str, Any], out_path: str, steps: int = GIF_STEPS, device: str = BASE_DEVICE):
    """
    Генерим красивую гифку нашим улучшенным рендером.
    Сохраняем прямо в out_path (например, .../episode.gif).
    """
    # имя для подписи на кадрах берём из имени эксперимента, если есть
    scen_name = os.path.splitext(os.path.basename(out_path))[0]
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)

    # Можно немного «поджать» масштаб для быстрой генерации (правь под себя):
    scale = 10         # пикселей на клетку в основной карте
    mini_scale = 5     # пикселей на клетку на мини-картах
    fps = 18
    grid_every = 8     # шаг сетки (0 — отключить)
    draw_fovs = True

    # ВАЖНО: simulate_ideal_gif сам всё отрисует (без модели, т.к. «идеальные» клипы управляются оракулом)
    # Если принципиально нужна именно политика — это отдельный режим, но сейчас используем «идеальные» клипы.
    simulate_ideal_gif(
        scen_name=scen_name,
        env_kwargs=env_kwargs,
        out_dir=out_dir,
        steps_limit=steps,
        seed=12345,
        scale=scale,
        mini_scale=mini_scale,
        fps=fps,
        show_grid_every=grid_every,
        draw_fovs=draw_fovs,
    )

    # simulate_ideal_gif сохраняет GIF как <out_dir>/<scen_name>.gif.
    # Если имя не совпало с желаемым out_path — просто переименуем:
    produced = os.path.join(out_dir, f"{scen_name}.gif")
    if produced != out_path and os.path.exists(produced):
        try:
            os.replace(produced, out_path)
        except Exception:
            shutil.copy2(produced, out_path)
    

def zip_dir(dir_path: str, zip_path: str):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(dir_path):
            for name in files:
                full = os.path.join(root, name)
                rel = os.path.relpath(full, dir_path)
                zf.write(full, rel)


# ========== Эксперимент ==========
@dataclass
class Experiment:
    name: str
    env_kwargs: Dict[str, Any]
    train_timesteps: int
    n_envs: int = BASE_N_ENVS
    n_steps: int = BASE_N_STEPS
    n_epochs: int = BASE_N_EPOCHS
    batch_size: int = BASE_N_ENVS * BASE_N_STEPS
    device: str = BASE_DEVICE


# ========== Конструктор параметров среды ==========

def base_param_space() -> List[Dict[str, Any]]:
    """Богатая сетка параметров для разнообразия сценариев."""
    grid_sizes          = [48, 64, 80]
    view_distances      = [4, 6, 8, 10, 12]
    view_angles         = [45, 60, 75, 90, 120]
    num_blues           = [1, 2, 3, 4]
    flee_flags          = [True, False]
    patrol_noises       = [0.0, 15.0, 30.0, 45.0]
    dyn_obst_steps      = [0, 50, 100, 150]
    obstacles_density   = [0.1, 0.2, 0.35, 0.5, 0.7]

    use_helpers         = [True, False]
    num_helpers         = [0, 1, 2, 3]    # будет учтён только если use_helpers=True
    use_last_seen       = [True, False]
    use_obstacle_memory = [True, False]
    use_detection_timer = [True, False]

    # базовые веса награды; потом будем в абляциях занулять
    reward_weight_sets = [
        dict(),  # default
        dict(looking_bonus=0.0),
        dict(proximity=0.5, step_penalty=-0.02),
        dict(explore_new=0.3, revisit_penalty=-0.02),
        dict(collision_penalty=-0.8),
    ]

    combos: List[Dict[str, Any]] = []
    for gs in grid_sizes:
        for vd in view_distances:
            for va in view_angles:
                for nb in num_blues:
                    for flee in flee_flags:
                        for pn in patrol_noises:
                            for dos in dyn_obst_steps:
                                for dens in obstacles_density:
                                    for helpers in use_helpers:
                                        for nh in num_helpers:
                                            if not helpers and nh != 0:
                                                continue
                                            for lseen in use_last_seen:
                                                for omem in use_obstacle_memory:
                                                    for dtimer in use_detection_timer:
                                                        for rw in reward_weight_sets:
                                                            combos.append(dict(
                                                                grid_size=gs,
                                                                view_distance=vd,
                                                                view_angle=va,
                                                                max_steps=2000,
                                                                num_blue=nb,
                                                                blue_flee_on_seen=flee,
                                                                patrol_noise=pn,
                                                                dynamic_obstacles_steps=dos,
                                                                obstacles_density=dens,
                                                                use_helpers=helpers,
                                                                num_helpers=nh,
                                                                use_last_seen=lseen,
                                                                use_obstacle_memory=omem,
                                                                use_detection_timer=dtimer,
                                                                reward_weights=rw if rw else None,
                                                            ))
    return combos


# ========== Абляции ==========
ABLATION_KEYS = [
    "use_helpers",
    "use_last_seen",
    "use_obstacle_memory",
    "use_detection_timer",
    # абляции по reward shaping:
    "rw_looking_bonus",
    "rw_proximity",
    "rw_explore_new",
]

def apply_ablation(env_kwargs: Dict[str, Any], abl: Tuple[str, ...]) -> Dict[str, Any]:
    kw = dict(env_kwargs)

    # аккуратно достаем веса (могут быть None)
    rw_src = kw.get("reward_weights") or {}

    # базовый набор с дефолтами
    rw = dict(
        discover_bonus=rw_src.get("discover_bonus", 0.0),
        proximity=rw_src.get("proximity", 1.0),
        looking_bonus=rw_src.get("looking_bonus", 0.5),
        collision_penalty=rw_src.get("collision_penalty", -0.5),
        step_penalty=rw_src.get("step_penalty", -0.01),
        explore_new=rw_src.get("explore_new", 0.5),
        revisit_penalty=rw_src.get("revisit_penalty", -0.01),
        explore_step_penalty=rw_src.get("explore_step_penalty", -0.005),
    )

    # применяем абляции
    for a in abl:
        if a == "use_helpers":
            kw["use_helpers"] = False
            kw["num_helpers"] = 0
        elif a == "use_last_seen":
            kw["use_last_seen"] = False
        elif a == "use_obstacle_memory":
            kw["use_obstacle_memory"] = False
        elif a == "use_detection_timer":
            kw["use_detection_timer"] = False
        elif a == "rw_looking_bonus":
            rw["looking_bonus"] = 0.0
        elif a == "rw_proximity":
            rw["proximity"] = 0.0
        elif a == "rw_explore_new":
            rw["explore_new"] = 0.0

    kw["reward_weights"] = rw
    return kw


# ========== Оценка сложности и длина обучения ==========
def difficulty_score(kw: Dict[str, Any]) -> float:
    """
    Грубая эвристика сложности сценария: выше — труднее.
    """
    score = 0.0
    # больше синих — сложнее
    score += 0.7 * (kw["num_blue"] - 1)
    # меньше обзор — сложнее
    score += (8 - min(kw["view_distance"], 8)) * 0.2
    score += (90 - min(kw["view_angle"], 90)) / 90 * 0.5
    # лабиринтность
    score += kw["obstacles_density"] * 1.0
    # динамические препятствия
    if kw["dynamic_obstacles_steps"] > 0:
        score += 0.5
    # если отключили полезные механики — сложнее
    if not kw["use_helpers"]:
        score += 0.4
    if not kw["use_last_seen"]:
        score += 0.3
    if not kw["use_obstacle_memory"]:
        score += 0.2
    # шум патруля — менее предсказуемо
    score += kw["patrol_noise"] / 90.0
    return max(0.0, score)

def train_steps_for(kw: Dict[str, Any]) -> int:
    """
    Переменная длина обучения: 200k..900k в зависимости от сложности.
    """
    base = 200_000
    extra = int(700_000 * min(1.0, difficulty_score(kw) / 3.0))
    return base + extra


# ========== Построение набора экспериментов ==========
def sample_experiments(max_n=MAX_EXPERIMENTS) -> List[Experiment]:
    random.seed(RANDOM_SEED)
    combos = base_param_space()

    # 1) случайные базовые сценарии
    picked = random.sample(combos, k=min(max_n // 2, len(combos)))

    # 2) абляции: одиночные и парные
    ablated: List[Dict[str, Any]] = []
    while len(ablated) < max_n - len(picked):
        base_kw = random.choice(combos)
        # выбираем 1 или 2 абляции
        k = random.choice([1, 2])
        abls = tuple(random.sample(ABLATION_KEYS, k=k))
        ablated_kw = apply_ablation(base_kw, abls)
        # избегаем дублей по сигнатуре
        sig = (base_kw["grid_size"], base_kw["view_distance"], base_kw["view_angle"],
               base_kw["num_blue"], base_kw["blue_flee_on_seen"], tuple(sorted(abls)))
        ablated.append((sig, ablated_kw, abls))

    # собираем итоговый список
    exps: List[Experiment] = []
    # базовые
    for i, kw in enumerate(picked):
        steps = train_steps_for(kw)
        name = (
            f"b{i:03d}_gs{kw['grid_size']}_vd{kw['view_distance']}_va{kw['view_angle']}"
            f"_nb{kw['num_blue']}_f{int(kw['blue_flee_on_seen'])}_pn{int(kw['patrol_noise'])}"
            f"_dos{kw['dynamic_obstacles_steps']}_dens{int(kw['obstacles_density']*100)}"
            f"_steps{steps//1000}k"
        )
        exps.append(Experiment(name=name, env_kwargs=kw, train_timesteps=steps))

    # абляционные
    for j, (_, kw, abls) in enumerate(ablated[: max_n - len(exps)]):
        steps = train_steps_for(kw)
        abl_tag = "x".join([a.replace("rw_", "") for a in abls])
        name = (
            f"a{j:03d}_{abl_tag}_gs{kw['grid_size']}_vd{kw['view_distance']}_va{kw['view_angle']}"
            f"_nb{kw['num_blue']}_pn{int(kw['patrol_noise'])}_dens{int(kw['obstacles_density']*100)}"
            f"_steps{steps//1000}k"
        )
        exps.append(Experiment(name=name, env_kwargs=kw, train_timesteps=steps))
    return exps


# ========== Один прогон эксперимента ==========
def run_one_experiment(exp: Experiment, out_root: str = OUT_ROOT) -> Tuple[str, str]:
    os.makedirs(out_root, exist_ok=True)
    run_dir = os.path.join(out_root, exp.name)
    ensure_clean_dir(run_dir)
    tb_dir = os.path.join(run_dir, "tb")
    os.makedirs(tb_dir, exist_ok=True)

    # векторная среда
    env_fns = [make_env_fn(1000, i, exp.env_kwargs) for i in range(exp.n_envs)]
    env = SubprocVecEnv(env_fns, start_method="spawn")
    env = VecMonitor(env)

    model = PPO(
        "CnnPolicy",
        env,
        verbose=0,
        device=exp.device,
        tensorboard_log=tb_dir,
        n_steps=exp.n_steps,
        batch_size=exp.batch_size,
        n_epochs=exp.n_epochs,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.03,
    )

    ckpt = CheckpointCallback(save_freq=(exp.n_envs * exp.n_steps) * 10, save_path=run_dir, name_prefix="ckpt")
    eval_env = RedAndBlueEnv(**exp.env_kwargs, seed=9999)
    eval_cb = EvalCallback(eval_env, best_model_save_path=run_dir, n_eval_episodes=8,
                           eval_freq=(exp.n_envs * exp.n_steps) * 5, deterministic=True, render=False)
    ep_logger = VecEpisodeLogger(check_freq=20, log_dir=run_dir, verbose=0)

    model.learn(total_timesteps=exp.train_timesteps, callback=[ckpt, eval_cb, ep_logger])
    model.save(os.path.join(run_dir, "final_model"))

    # оценка (3 эпизода)
    stats = []
    for i in range(3):
        env_eval = RedAndBlueEnv(**exp.env_kwargs, seed=2024 + i)
        obs, info = env_eval.reset()
        terminated = truncated = False
        total_r = 0.0
        steps = 0
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env_eval.step(action)
            total_r += reward
            steps += 1
            if steps >= 2000:
                break
        stats.append(dict(
            episode=i,
            reward=float(total_r),
            is_win=bool(info.get("is_win", False)),
            is_loss=bool(info.get("is_loss", False)),
            steps=int(info.get("episode_length", steps)),
        ))

    # GIF
    gif_path = os.path.join(run_dir, "episode.gif")
    try:
        record_gif(model, exp.env_kwargs, gif_path, steps=GIF_STEPS, device=exp.device)
    except Exception as e:
        print(f"[GIF] skipped: {e}")

    # CSV + JSON
    csv_path = os.path.join(run_dir, "eval_stats.csv")
    with open(csv_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(stats[0].keys()))
        wr.writeheader()
        wr.writerows(stats)
    with open(os.path.join(run_dir, "experiment.json"), "w") as f:
        json.dump(asdict(exp), f, indent=2)

    # ZIP и отправка
    zip_path = os.path.join(OUT_ROOT, f"{exp.name}.zip")
    zip_dir(run_dir, zip_path)

    send_file_telegram(zip_path, caption=f"ZIP: {exp.name}")
    send_file_telegram(csv_path, caption=f"CSV: {exp.name}")
    if os.path.exists(gif_path):
        send_file_telegram(gif_path, caption=f"GIF: {exp.name}")

    return run_dir, zip_path


# ========== Сводка ==========
def draw_summary_plot(rows: List[Dict[str, Any]], out_path: str):
    rows_sorted = sorted(rows, key=lambda r: r["win_rate"], reverse=True)
    names = [r["name"] for r in rows_sorted]
    win_rates = [r["win_rate"] for r in rows_sorted]
    plt.figure(figsize=(12, max(4, len(names) * 0.12)))
    plt.barh(names, win_rates)
    plt.xlabel("Win Rate")
    plt.title("Experiments Win Rate")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ========== main ==========
def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    experiments = sample_experiments(MAX_EXPERIMENTS)
    print(f"Total planned experiments: {len(experiments)}")

    all_rows = []
    for idx, exp in enumerate(experiments, 1):
        print(f"\n=== [{idx}/{len(experiments)}] {exp.name} ({exp.train_timesteps} steps) ===")
        run_dir, zip_path = run_one_experiment(exp, out_root=OUT_ROOT)

        csv_path = os.path.join(run_dir, "eval_stats.csv")
        data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding=None)
        if data.ndim == 0:
            data = np.array([data])
        rew_mean = float(np.mean(data["reward"]))
        win_rate = float(np.mean(data["is_win"]))
        steps_mean = float(np.mean(data["steps"]))
        row = dict(name=exp.name, rew_mean=rew_mean, win_rate=win_rate, steps_mean=steps_mean)
        all_rows.append(row)

        # каждые 10 экспериментов — промежуточная сводка
        if idx % 10 == 0:
            tmp_sum = os.path.join(OUT_ROOT, f"summary_{idx}.csv")
            with open(tmp_sum, "w", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=["name", "rew_mean", "win_rate", "steps_mean"])
                wr.writeheader()
                wr.writerows(all_rows)
            send_file_telegram(tmp_sum, caption=f"SUMMARY {idx}/{len(experiments)}")

    # финал
    summary_csv = os.path.join(OUT_ROOT, "summary.csv")
    with open(summary_csv, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["name", "rew_mean", "win_rate", "steps_mean"])
        wr.writeheader()
        wr.writerows(all_rows)
    summary_png = os.path.join(OUT_ROOT, "summary_winrate.png")
    draw_summary_plot(all_rows, summary_png)
    send_file_telegram(summary_csv, caption="EXPERIMENTS SUMMARY")
    send_file_telegram(summary_png, caption="EXPERIMENTS SUMMARY PLOT")


if __name__ == "__main__":
    os.environ.setdefault("MPLBACKEND", "Agg")
    main()
