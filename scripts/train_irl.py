import numpy as np
#from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env
from imitation.data.wrappers import RolloutInfoWrapper
from scm_irl.env.scm_irl_env import ScmIrlEnv
from scm_irl.utils.process_scenario import Scenario
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

gym.register(
    id='ScmIrl-v0',
    entry_point='scm_irl.env:ScmIrlEnv',
    max_episode_steps=1000,
)


@hydra.main(config_path="../scm_irl/conf", config_name="train_irl")
def train(cfg: DictConfig) -> None:

    date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    model_name = f"{cfg.model}_{cfg.irl_algo}_{cfg.lerner_algo}_{date_time}"

    output_path = os.path.join("../outputs", model_name)
    os.makedirs(output_path, exist_ok=True)

    data_path = os.path.join(utils.get_original_cwd(), cfg.scenarios_path)
    
    dict_scenarios = {}

    wandb_cfg = OmegaConf.to_container(cfg, resolve=True)
    wandb.init(
        name=model_name,
        dir=output_path,
        config=wandb_cfg,
        sync_tensorboard=True,  # automatically upload SB3's tensorboard metrics to W&B
        project="scm_irl",
        monitor_gym=True,       # automatically upload gym environments' videos
        save_code=True,
    )

    if not cfg.multi_scenario:
        path_scenario = os.path.join(data_path, f"scenario_{cfg.scenario_id}")
        scenario = Scenario(cfg, path_scenario)
        if cfg.mmsi in scenario.get_valid_vessels():
            dict_scenarios[f"{cfg.scenario_id}_{cfg.mmsi}"] = scenario
        else:
            for mmsi in scenario.get_valid_vessels():
                dict_scenarios[f"{cfg.scenario_id}_{mmsi}"] = scenario

    else:
        for dir in os.listdir(data_path):
            if os.path.isdir(os.path.join(data_path, dir)):
                path_scenario = os.path.join(data_path, dir)
                scenario_id = dir.split("_")[-1]
                
                scenario = Scenario(cfg, path_scenario)
                for mmsi in scenario.get_valid_vessels():
                    dict_scenarios[f"{scenario_id}_{mmsi}"] = scenario
            #print(len(dict_scenarios))
            if len(dict_scenarios) >= cfg.num_envs:
                break


    def make_env(mode = '', cfg_env = None, rank = 0, dict_scenarios = {}, seed=0, list_scenarios = [], video_enable = False):
        def _init():
            scenario_id = list_scenarios[rank]
            scenario = dict_scenarios[scenario_id]
            mmsi = int(scenario_id.split("_")[-1])
            env = ScmIrlEnv(cfg_env, scenario, mmsi=mmsi, awareness_zone = [200, 500, 200, 200], render_mode="rgb_array")

            print(env)
            #env = FlatObservationWrapper(env)
            #print(env.observation_space)
            if rank < 5 and video_enable:  # only add the RecordVideo wrapper for the first environment
                env = gym.wrappers.RecordVideo(env, name_prefix=f"{mode}_{rank}", video_folder=f"{output_path}/videos_{mode}")  # record videos
            env = gym.wrappers.RecordEpisodeStatistics(env)  # rec0ord stats such as returns
            return env
        return _init

    list_vessels_scenarios = list(dict_scenarios.keys())
    num_scenarions_mmsi = len(list_vessels_scenarios)

    if cfg.num_envs > num_scenarions_mmsi:
        raise ValueError("The number of environments is greater than the number of possible scenarios")
    
    cfg_env = cfg.copy()
    cfg_env['env']['copy_expert'] = True
    env = DummyVecEnv([make_env(mode='rollout', cfg_env=cfg, rank=i, seed=SEED,
                                 dict_scenarios=dict_scenarios, 
                                 list_scenarios=list_vessels_scenarios) for i in range(cfg.num_envs)])
        
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

    print("############# Load Policy")
    expert = load_policy(
        "copy_action",
        organization="scm",
        env_name="ScmIrl-v0",
        venv=env,
    )


    print("############# rollouts")
    rollouts = rollout.rollout(
        expert,
        env,
        sample_until=rollout.make_sample_until(min_timesteps=None, min_episodes=cfg['irl_params']['min_expert_demos']),
        unwrap=False,
        rng=np.random.default_rng(SEED),
        exclude_infos=True,
        verbose=True,
    )
    # print("Rollouts")
    # print(rollouts)

    print("############# learner")
    learner = PPO(
        env=env,
        policy=MlpPolicy,
        batch_size=cfg['irl_params']['learner_batch_size'],
        ent_coef=0.0,
        learning_rate=cfg['irl_params']['learner_lr'],
        gamma=0.95,
        n_epochs=5,
        seed=SEED,
        tensorboard_log=f"{output_path}/algo",
        verbose=1,
    )

    print("############# reward_net")
    reward_net = BasicRewardNet(
        observation_space=env.observation_space,
        action_space=env.action_space,
        normalize_input_layer=RunningNorm,
    )

    print("############# gail_trainer")
    gail_trainer = GAIL(
        demonstrations=rollouts,
        demo_batch_size=512,
        gen_replay_buffer_capacity=512,
        n_disc_updates_per_round=8,
        venv=env,
        gen_algo=learner,
        reward_net=reward_net,
        allow_variable_horizon=True,
    )

    # cfg_env = cfg.copy()
    # cfg_env['env']['copy_expert'] = False
    # env = DummyVecEnv([make_env(mode='pre', cfg_env=cfg_env, rank=i, seed=SEED,
    #                              dict_scenarios=dict_scenarios, 
    #                              list_scenarios=list_vessels_scenarios) for i in range(cfg.num_envs)])

    # print("############# evaluate_policy")
    # learner_rewards_before_training, _ = evaluate_policy(
    #     learner, env, 5, return_episode_rewards=True)
    
    # print("Rewards before training")
    # print(learner_rewards_before_training)

    print("############# Train")
    gail_trainer.train(cfg['irl_params']['total_timesteps'])



    cfg_env = cfg.copy()
    cfg_env['env']['copy_expert'] = False
    env_post = DummyVecEnv([make_env(mode='post', cfg_env=cfg_env, rank=i, seed=SEED,
                                 dict_scenarios=dict_scenarios, 
                                 list_scenarios=list_vessels_scenarios, video_enable = True) for i in range(cfg.num_envs)])

    learner_rewards_after_training, _ = evaluate_policy(
        learner, env_post, 5, return_episode_rewards=True)
    
    print("Rewards after training")
    print(learner_rewards_after_training)

    wandb.log({"Rewards after training": learner_rewards_after_training})

    # At the end of your script, optionally
    wandb.finish()


if __name__ == "__main__":
    train()

