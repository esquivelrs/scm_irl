from scm_irl.env.scm_irl_env import ScmIrlEnv
from sllib.conversions.geo_conversions import north_east_to_lat_lon, mps2knots, lat_lon_to_north_east
import numpy as np
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


import gymnasium as gym
# Check if the environment is registered

gym.register(
    id='ScmIrl-v0',
    entry_point='scm_irl.env:ScmIrlEnv',
    max_episode_steps=1000,
)


def make_env():
    #env = gym.make("CartPole-v1", render_mode="rgb_array")
    path = "/zhome/11/1/193832/resquivel/scm_irl/data/raw/scenario_2a66ceaf61"
    env = ScmIrlEnv(path, mmsi=215811000, awareness_zone = [200, 500, 200, 200], start_time_reference=1577905000.0, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, f"videos")  # record videos
    env = gym.wrappers.RecordEpisodeStatistics(env)  # record stats such as returns
    return env

config = {
    "policy": 'MultiInputPolicy',
    "total_timesteps": 400000,
    "env_name": "ScmIrl-v0",
}

wandb.init(
    config=config,
    sync_tensorboard=True,  # automatically upload SB3's tensorboard metrics to W&B
    project="scmirl",
    monitor_gym=True,       # automatically upload gym environements' videos
    save_code=True,
)

env = DummyVecEnv([make_env])
model = PPO(config['policy'], env, verbose=1, tensorboard_log=f"runs/ppo")
model.learn(total_timesteps=config['total_timesteps'])
wandb.finish()

