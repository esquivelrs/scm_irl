import os
import datetime
import numpy as np
from omegaconf import DictConfig
import hydra
from hydra import utils
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from stable_baselines3 import PPO, SAC, DDPG, TD3, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
import wandb
from wandb.integration.sb3 import WandbCallback
from scm_irl.env.scm_irl_env import ScmIrlEnv
from sllib.conversions.geo_conversions import north_east_to_lat_lon, mps2knots, lat_lon_to_north_east
import gymnasium as gym
from scm_irl.utils.env_wrappers import FlatObservationWrapper

import gymnasium as gym
# Check if the environment is registered

gym.register(
    id='ScmIrl-v0',
    entry_point='scm_irl.env:ScmIrlEnv',
    max_episode_steps=1000,
)


@hydra.main(config_path="../scm_irl/conf", config_name="train_rl")
def train(cfg: DictConfig) -> None:

    date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    model_name = f"scm_2a66ceaf61_{date_time}"

    output_path = os.path.join(utils.get_original_cwd(), "../outputs", model_name)
    os.makedirs(output_path, exist_ok=True)

    def make_env(env_id, rank, seed=0):
        def _init():
            path = "../data/raw/scenario_2a66ceaf61"
            path = os.path.join(utils.get_original_cwd(), path)
            env = ScmIrlEnv(cfg, path, mmsi=215811000, awareness_zone = [200, 500, 200, 200], start_time_reference=1577905000.0, render_mode="rgb_array")
            #env = FlatObservationWrapper(env)
            print(env.observation_space)
            if rank == 0:  # only add the RecordVideo wrapper for the first environment
                env = gym.wrappers.RecordVideo(env, f"{output_path}/videos")  # record videos
            env = gym.wrappers.RecordEpisodeStatistics(env)  # record stats such as returns
            return env
        return _init

    wandb_cfg = OmegaConf.to_container(cfg, resolve=True)
    wandb.init(
        name=model_name,
        dir=output_path,
        config=wandb_cfg,
        sync_tensorboard=True,  # automatically upload SB3's tensorboard metrics to W&B
        project="scmirl",
        monitor_gym=True,       # automatically upload gym environments' videos
        save_code=True,
    )

    num_envs = 1
    env = DummyVecEnv([make_env(cfg.env_name, i) for i in range(num_envs)])

    model = A2C(cfg.policy, env, verbose=1, tensorboard_log=f"runs/ppo")
    model.learn(total_timesteps=cfg.total_timesteps)

    # save model
    model.save(f"{output_path}/ppo_ais")

    wandb.finish()

if __name__ == "__main__":
    train()