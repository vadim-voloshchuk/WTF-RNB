import os
import time
from stable_baselines3 import PPO
from environment.red_and_blue_env_cnn import RedAndBlueEnv


def run_episode(model_path: str, render: bool = False, device: str = "cuda"):
    """model_path без расширения .zip (SB3 сам справится)"""
    env = RedAndBlueEnv(grid_size=64, view_distance=8, view_angle=90, max_steps=2000, seed=777)
    model = PPO.load(model_path, device=device)

    obs, info = env.reset()
    terminated = False
    truncated = False
    total_r = 0.0

    while not (terminated or truncated):
        if render:
            env.render()
            time.sleep(0.01)
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_r += reward

    print(
        f"Done. R={total_r:.2f}, "
        f"is_win={info.get('is_win')}, is_loss={info.get('is_loss')}, "
        f"steps={info.get('episode_length')}"
    )
    return total_r, info


if __name__ == "__main__":
    # Автопоиск лучшей/финальной модели
    base = "runs/ppo_red_blue"
    candidates = ["best_model", "final_model"]

    model_path = None
    for name in candidates:
        if os.path.exists(os.path.join(base, name + ".zip")):
            model_path = os.path.join(base, name)
            break

    if model_path is None:
        raise FileNotFoundError(
            "Модель не найдена. Сначала запусти обучение (train.py), "
            "а потом попробуй снова."
        )

    print("Loading:", model_path)
    run_episode(model_path, render=False, device="cuda")
