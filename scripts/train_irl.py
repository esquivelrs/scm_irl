import numpy as np
#from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env
from imitation.data.wrappers import RolloutInfoWrapper
from scm_irl.env.scm_irl_env import ScmIrlEnv
from sllib.conversions.geo_conversions import north_east_to_lat_lon, mps2knots, lat_lon_to_north_east
import numpy as np
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.data import rollout
import datetime
import os

from omegaconf import DictConfig
import hydra
from hydra import utils
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from scm_irl.utils.env_wrappers import FlatObservationWrapper
from imitation.util import util
from imitation.policies.serialize import policy_registry

from imitation.algorithms.adversarial.gail import GAIL
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy


import gymnasium as gym
# Check if the environment is registered



SEED = 42



@hydra.main(config_path="../scm_irl/conf", config_name="train_irl")
def train(cfg: DictConfig) -> None:

    date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    model_name = f"scm_2a66ceaf61_{date_time}"

    output_path = os.path.join("../outputs", model_name)
    os.makedirs(output_path, exist_ok=True)


    def make_env(env_id, rank, seed=0):
        def _init():
            path = "../data/raw/scenario_2a66ceaf61"
            path = os.path.join(utils.get_original_cwd(), path)
            env = ScmIrlEnv(cfg, path, mmsi=215811000, awareness_zone = [200, 500, 200, 200], start_time_reference=1577905000.0, end_time_override = 1577905020.0, render_mode="rgb_array")
            
            #env = gym.wrappers.FlattenObservation(env)
            
            #env = FlatObservationWrapper(env)
            # print("Observation space")
            # print(env.observation_space)
            # if rank == 0:  # only add the RecordVideo wrapper for the first environment
            #     env = gym.wrappers.RecordVideo(env, f"{output_path}/videos")  # record videos
            #env = gym.wrappers.RecordEpisodeStatistics(env)  # record stats such as returns
            return env
        return _init


    num_envs = 1
    #env = make_env("ScmIrl-v0", rank=0, seed=SEED)

    
    #env = DummyVecEnv([make_env("ScmIrl-v0", i) for i in range(num_envs)])

    path = "../data/raw/scenario_2a66ceaf61"
    path = os.path.join(utils.get_original_cwd(), path)
    # env = ScmIrlEnv(cfg, path, mmsi=215811000, awareness_zone = [200, 500, 200, 200], start_time_reference=1577905000.0, end_time_override = 1577905020.0, render_mode="rgb_array")

    gym.register(
        id='ScmIrl-v0',
        entry_point='scm_irl.env.scm_irl_env:ScmIrlEnv',
        max_episode_steps=2000,
        kwargs={"cfg": cfg, 
                 "scenario_path": path,
                 "mmsi": 215811000, 
                 "awareness_zone": [200, 500, 200, 200], 
                 #"start_time_reference": 1577905000.0, 
                 #"end_time_override": 1577905020.0, 
                 "render_mode": "rgb_array"}
                 )

    
    env = make_vec_env("ScmIrl-v0", n_envs=num_envs, 
                       rng=np.random.default_rng(SEED))
    
    print("############# Registered")


    def load_policy(policy_name, organization, env_name, venv):
        def policy_fn(obs, state, dones):
            actions = []
            states = []
            for env in venv.envs:
                # Get the current timestep
                timestep = env.unwrapped.timestep

                # Get the action from the vessel
                action = env.unwrapped.get_action_from_vessel(timestep)
                actions.append(action)
                states.append(None)
            # print("Action")
            # timestep = env.timestep

            # # Get the action from the vessel
            # action = env.get_action_from_vessel(timestep)
            # actions.append(action)
            # states.append(None)

            #print(actions)
            return actions, states

        return policy_fn

    #policy_registry.register("my-policy", load_policy)

    expert = load_policy(
        "copy_action",
        organization="scm",
        env_name="ScmIrl-v0",
        venv=env,
    )



    rollouts = rollout.rollout(
        expert,
        env,
        sample_until=rollout.make_sample_until(min_timesteps=None, min_episodes=20),
        unwrap=False,
        rng=np.random.default_rng(SEED),
        exclude_infos=True,
        verbose=True,
    )
    # print("Rollouts")
    # print(rollouts)


    learner = PPO(
        env=env,
        policy=MlpPolicy,
        batch_size=64,
        ent_coef=0.0,
        learning_rate=0.0004,
        gamma=0.95,
        n_epochs=5,
        seed=SEED,
    )

    reward_net = BasicRewardNet(
        observation_space=env.observation_space,
        action_space=env.action_space,
        normalize_input_layer=RunningNorm,
    )
    gail_trainer = GAIL(
        demonstrations=rollouts,
        demo_batch_size=20,
        gen_replay_buffer_capacity=512,
        n_disc_updates_per_round=8,
        venv=env,
        gen_algo=learner,
        reward_net=reward_net,
    )
    learner_rewards_before_training, _ = evaluate_policy(
        learner, env, 100, return_episode_rewards=True)
    
    print("Rewards before training")
    print(learner_rewards_before_training)

    gail_trainer.train(10_000)

    learner_rewards_after_training, _ = evaluate_policy(
        learner, env, 100, return_episode_rewards=True)
    
    print("Rewards after training")
    print(learner_rewards_after_training)


if __name__ == "__main__":
    train()

