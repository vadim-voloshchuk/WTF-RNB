import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, ProgressBarCallback
from callbacks import VecEpisodeLogger
from make_envs import make_env, seed_everything
from environment.red_and_blue_env_cnn import RedAndBlueEnv


def main():
    seed = 42
    seed_everything(seed)

    log_dir = "runs/ppo_red_blue"
    os.makedirs(log_dir, exist_ok=True)

    # ── Векторная среда ──────────────────────────────────────────────────────────
    n_envs = 8  # начни с 8; потом поднимай до 32/64
    env = SubprocVecEnv([make_env(seed, i) for i in range(n_envs)], start_method="spawn")
    env = VecMonitor(env)  # один монитор на весь вектор
    print("num_envs:", env.num_envs)  # sanity-check

    # ── Одиночная среда для EVAL (создаём ДО коллбэков!) ───────────────────────
    eval_env = RedAndBlueEnv(grid_size=64, view_distance=8, view_angle=90, max_steps=2000, seed=seed + 123)

    # ── Гиперпараметры PPO ──────────────────────────────────────────────────────
    # rollout = n_envs * n_steps; batch_size делаем кратным rollout
    n_steps = 128
    batch_size = n_envs * n_steps  # 8 * 128 = 1024
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        device="cuda",  # или "cpu"
        tensorboard_log=log_dir,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=6,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.03,
    )

    # ── Коллбэки ────────────────────────────────────────────────────────────────
    ckpt = CheckpointCallback(
        save_freq=(n_envs * n_steps) * 10,  # каждые ~10 rollout-циклов
        save_path=log_dir,
        name_prefix="ckpt",
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        n_eval_episodes=10,
        eval_freq=(n_envs * n_steps) * 5,
        deterministic=True,
        render=False,
    )

    ep_logger = VecEpisodeLogger(check_freq=20, log_dir=log_dir, verbose=1)

    # Progress bar (если нет tqdm/rich — тихо отключим)
    try:
        pbar = ProgressBarCallback()
        callbacks = [ckpt, eval_cb, ep_logger, pbar]
    except Exception:
        callbacks = [ckpt, eval_cb, ep_logger]

    # ── Обучение ────────────────────────────────────────────────────────────────
    total_timesteps = 3_000_000
    model.learn(total_timesteps=total_timesteps, callback=callbacks)

    # ── Сохранение финальной модели ─────────────────────────────────────────────
    model.save(os.path.join(log_dir, "final_model"))
    print("Saved final model to:", os.path.join(log_dir, "final_model") + ".zip")


if __name__ == "__main__":
    # На всякий случай: если хочешь форснуть неинтерактивный бекенд
    # через переменную окружения, сделай в шелле:
    #   export MPLBACKEND=Agg
    main()
