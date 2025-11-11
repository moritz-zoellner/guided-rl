import os
import argparse
from datetime import datetime
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
    BaseCallback,
)
from stable_baselines3.common.vec_env import DummyVecEnv

import wandb
from wandb.integration.sb3 import WandbCallback

def make_train_env(env_id: str):
    return Monitor(gym.make(env_id))

def make_eval_env(env_id: str, video_dir: str):
    base = gym.make(env_id, render_mode="rgb_array")
    wrapped = RecordVideo(
        base,
        video_folder=video_dir,
        episode_trigger=lambda ep: True,  # record every episode during eval pass
        name_prefix="humanoid_eval",
    )
    return Monitor(wrapped)

class ConsoleLogCallback(BaseCallback):
    """Aggregate recent rewards and report to stdout without extra env rollouts."""

    def __init__(self, window: int = 500):
        super().__init__()
        self.window = max(window, 1)
        self._recent_rewards: list[float] = []
        self.print_count = 1

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            episode_info = info.get("episode")
            if episode_info is None:
                continue
            reward = episode_info.get("r")
            if reward is None:
                continue
            self._recent_rewards.append(float(reward))
            timestep = self.model.num_timesteps
            if timestep > self.window * self.print_count:
                avg_reward = sum(self._recent_rewards) / len(self._recent_rewards)
                print(
                    f"Time_step: {timestep}, "
                    f"MeanReward[{len(self._recent_rewards) if self.window > 1 else 1}]: {avg_reward:.2f}"
                )
                self._recent_rewards = []
                self.print_count+=1
        return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="Humanoid-v4")
    parser.add_argument("--total_timesteps", type=int, default=200_000)
    parser.add_argument("--checkpoint_interval", type=int, default=40_000)
    parser.add_argument("--eval_freq", type=int, default=40_000)
    parser.add_argument("--eval_episodes", type=int, default=1)
    parser.add_argument("--logdir", type=str, default="runs")
    parser.add_argument("--models_dir", type=str, default="checkpoints")
    parser.add_argument("--videos_dir", type=str, default="videos")
    parser.add_argument("--print_window", type=int, default=500)
    parser.add_argument("--wandb_project", type=str, default="Guided_RL")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(args.logdir, timestamp)
    models_dir = os.path.join(run_dir, args.models_dir)
    videos_dir = os.path.join(run_dir, args.videos_dir)
    eval_log_dir = os.path.join(run_dir, "eval")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(eval_log_dir, exist_ok=True)

    # ---- Training env (no rendering; fast) ----
    train_env = DummyVecEnv([lambda: make_train_env(args.env_id)])

    # ---- Eval env (records video during each eval) ----
    eval_env_raw = make_eval_env(args.env_id, videos_dir)
    eval_env = DummyVecEnv([lambda: eval_env_raw])

    # ---- Model ----
    model = PPO(
        "MlpPolicy",
        train_env,
        device="auto",                 # CPU on laptop, GPU on cluster
        verbose=0,
    )

    # ---- Callbacks ----
    checkpoint_cb = CheckpointCallback(
        save_freq=args.checkpoint_interval,
        save_path=models_dir,
        name_prefix=args.env_id,
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=models_dir,    # also saves best model
        log_path=eval_log_dir,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,  # rendering handled by RecordVideo wrapper
    )


    console_cb = ConsoleLogCallback(window=args.print_window)
    callbacks_list = [checkpoint_cb, eval_cb, console_cb]

    wandb_run = None
    if args.wandb_project:
        api_key = os.environ.get("WANDB_API_KEY")
        if api_key:
            wandb.login(key=api_key, relogin=True)
        else:
            print("⚠️ Set WANDB_API_KEY in your env for non-interactive wandb auth.")
            wandb.login()
        wandb_run = wandb.init(
            project=args.wandb_project,
            # entity=args.wandb_entity,
            name=args.wandb_run_name or timestamp,
            config={
                "env_id": args.env_id,
                "total_timesteps": args.total_timesteps,
                "checkpoint_interval": args.checkpoint_interval,
                "eval_freq": args.eval_freq,
                "eval_episodes": args.eval_episodes,
                "print_window": args.print_window,
            },
            monitor_gym=True,
            save_code=True,
        )
        callbacks_list.append(
            WandbCallback(model_save_path=None, gradient_save_freq=0, verbose=0)
        )

    callbacks = CallbackList(callbacks_list)

    # ---- Train ----
    model.learn(total_timesteps=args.total_timesteps, callback=callbacks)

    # ---- Final save ----
    final_path = os.path.join(models_dir, "final_policy")
    model.save(final_path)
    if wandb_run is not None:
        wandb_run.finish()
    print(f"Finished. Final model saved to: {final_path}")
    print(f"Run directory: {run_dir}")
    print(f"Videos saved in: {videos_dir}")
    print(f"Checkpoints in: {models_dir}")
    print(f" Eval logs in: {eval_log_dir}")



if __name__ == "__main__":
    main()
