# callbacks.py
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3.common.callbacks import BaseCallback

class VecEpisodeLogger(BaseCallback):
    def __init__(self, check_freq=20, log_dir="runs", verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.ep_rewards = []
        self.ep_lengths = []

    def _on_training_start(self) -> None:
        n_envs = getattr(self.training_env, "num_envs", 1)
        self._ret = np.zeros(n_envs, dtype=np.float32)
        self._len = np.zeros(n_envs, dtype=np.int32)

    def _on_step(self) -> bool:
        rewards = self.locals["rewards"]
        dones   = self.locals["dones"]
        infos   = self.locals["infos"]

        self._ret += rewards
        self._len += 1

        for i, d in enumerate(dones):
            if d:
                epi = infos[i].get("episode")
                if epi:
                    r = float(epi["r"]); l = int(epi["l"])
                else:
                    r = float(self._ret[i]); l = int(self._len[i])
                self.ep_rewards.append(r)
                self.ep_lengths.append(l)
                self._ret[i] = 0.0
                self._len[i] = 0

        if len(self.ep_rewards) > 0 and len(self.ep_rewards) % self.check_freq == 0:
            mean_r = float(np.mean(self.ep_rewards[-10:]))
            mean_l = float(np.mean(self.ep_lengths[-10:]))
            if self.verbose:
                print(f"[{len(self.ep_rewards)} eps] last={self.ep_rewards[-1]:.2f} mean10={mean_r:.2f} len10={mean_l:.1f}")
            try:
                self.logger.record("rollout/ep_rew_mean_last10", mean_r)
                self.logger.record("rollout/ep_len_mean_last10", mean_l)
            except Exception:
                pass
            self._save_plot()
        return True

    def _on_training_end(self) -> None:
        self._save_plot()

    def _save_plot(self):
        if not self.ep_rewards:
            return
        try:
            plt.figure(figsize=(12, 6))
            plt.plot(self.ep_rewards, label="Episode reward")
            if len(self.ep_rewards) >= 10:
                ma = np.convolve(self.ep_rewards, np.ones(10)/10, mode="valid")
                x = np.arange(10, len(self.ep_rewards) + 1)
                plt.plot(x, ma, label="Mean(10)")
            plt.xlabel("Episodes")
            plt.ylabel("Reward")
            plt.legend()
            plt.tight_layout()
            path = os.path.join(self.log_dir, "rewards_plot.png")
            plt.savefig(path)
            plt.close()
            if self.verbose:
                print(f"Saved: {path}")
        except Exception as e:
            if self.verbose:
                print(f"[plot skipped] {e}")
