from scm_irl.env.scm_irl_env import ScmIrlEnv
from sllib.conversions.geo_conversions import north_east_to_lat_lon, mps2knots, lat_lon_to_north_east
import numpy as np
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import datetime


import gymnasium as gym
# Check if the environment is registered

gym.register(
    id='ScmIrl-v0',
    entry_point='scm_irl.env:ScmIrlEnv',
    max_episode_steps=1000,
)


def make_env(env_id, rank, seed=0):
    def _init():
        path = "../data/raw/scenario_2a66ceaf61"
        env = ScmIrlEnv(path, mmsi=215811000, awareness_zone = [200, 500, 200, 200], start_time_reference=1577905000.0, render_mode="rgb_array")
        #env = gym.wrappers.RecordVideo(env, f"videos")  # record videos
        if rank == 0:  # only add the RecordVideo wrapper for the first environment
            env = gym.wrappers.RecordVideo(env, f"videos")  # record videos
        env = gym.wrappers.RecordEpisodeStatistics(env)  # record stats such as returns
        #env.seed(seed + rank)  # ensure each environment has a different seed
        return env
    return _init


config = {
    "policy": 'MultiInputPolicy',
    "total_timesteps": 400000,
    "env_name": "ScmIrl-v0",
}

date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
model = f"scm_2a66ceaf61_{date_time}"

wandb.init(
    name=date_time,
    dir=f"../outputs/{date_time}",
    config=config,
    sync_tensorboard=True,  # automatically upload SB3's tensorboard metrics to W&B
    project="scmirl",
    monitor_gym=True,       # automatically upload gym environments' videos
    save_code=True,
)

num_envs = 4
env = DummyVecEnv([make_env("ScmIrl-v0", i) for i in range(num_envs)])

model = PPO(config['policy'], env, verbose=1, tensorboard_log=f"runs/ppo")
model.learn(total_timesteps=config['total_timesteps'])

wandb.finish()

