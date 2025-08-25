# make_envs.py
from stable_baselines3.common.utils import set_random_seed
from environment.red_and_blue_env_cnn import RedAndBlueEnv

def make_env(seed: int, idx: int):
    def _init():
        env = RedAndBlueEnv(grid_size=64, view_distance=8, view_angle=90, max_steps=2000, seed=seed + idx)
        return env  # БЕЗ Monitor тут
    return _init

def seed_everything(seed: int):
    set_random_seed(seed)
