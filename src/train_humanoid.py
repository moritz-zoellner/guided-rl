import os
import argparse
from datetime import datetime
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv


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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="Humanoid-v4")
    parser.add_argument("--total_timesteps", type=int, default=2_000_000)
    parser.add_argument("--checkpoint_interval", type=int, default=200_000)
    parser.add_argument("--eval_freq", type=int, default=200_000)
    parser.add_argument("--eval_episodes", type=int, default=1)
    parser.add_argument("--logdir", type=str, default="runs")
    parser.add_argument("--models_dir", type=str, default="checkpoints")
    parser.add_argument("--videos_dir", type=str, default="videos")
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
        verbose=1,
    )

    # ---- Callbacks ----
    checkpoint_cb = CheckpointCallback(
        save_freq=args.checkpoint_interval,
        save_path=models_dir,
        name_prefix="humanoid_ppo",
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

    callbacks = CallbackList([checkpoint_cb, eval_cb])

    # ---- Train ----
    model.learn(total_timesteps=args.total_timesteps, callback=callbacks)

    # ---- Final save ----
    final_path = os.path.join(models_dir, "final_policy")
    model.save(final_path)
    print(f"✅ Finished. Final model saved to: {final_path}")
    print(f"📁 Run directory: {run_dir}")
    print(f"📼 Videos saved in: {videos_dir}")
    print(f"💾 Checkpoints in: {models_dir}")
    print(f"🪵 Eval logs in: {eval_log_dir}")


if __name__ == "__main__":
    main()
