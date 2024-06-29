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
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecVideoRecorder, VecNormalize
from imitation.data import rollout
import datetime
import os
import pathlib

from omegaconf import DictConfig
import hydra
from hydra import utils
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from scm_irl.utils.env_wrappers import FlatObservationWrapper, ResNetObservationWrapper
from imitation.util import util
from imitation.policies.serialize import policy_registry
from imitation.scripts.train_adversarial import save

from imitation.algorithms.adversarial.gail import GAIL
from imitation.algorithms.adversarial.airl import AIRL
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.util import logger
from imitation.data.wrappers import RolloutInfoWrapper
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.util import logger as imit_logger



import gymnasium as gym
# Check if the environment is registered



SEED = 42

gym.register(
    id='ScmIrl-v0',
    entry_point='scm_irl.env:ScmIrlEnv',
    max_episode_steps=1000,
)

def callback(round_num: int, /) -> None:
    if checkpoint_interval > 0 and  round_num % checkpoint_interval == 0:
        save(airl_trainer, pathlib.Path(f"checkpoints/checkpoint{round_num:05d}"))


@hydra.main(config_path="../scm_irl/conf", config_name="train_irl")
def train(cfg: DictConfig) -> None:

    output_dir = os.path.join(utils.get_original_cwd(), cfg.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    model_name = f"{cfg.model}_{cfg.irl_algo}_{cfg.lerner_algo}_{date_time}"

    data_path = os.path.join(utils.get_original_cwd(), cfg.scenarios_path)
    
    dict_scenarios = {}

    wandb_cfg = OmegaConf.to_container(cfg, resolve=True)
    wandb.init(
        name=model_name,
        dir=output_dir,
        config=wandb_cfg,
        sync_tensorboard=True,  # automatically upload SB3's tensorboard metrics to W&B
        tensorboard=True,       # enable tensorboard
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
            env = ScmIrlEnv(cfg_env, scenario, mmsi=mmsi, awareness_zone = cfg_env['env']['awareness_zone'],
                             render_mode="rgb_array", resolution=cfg_env['env']['resolution'])

            #print(env)
            if cfg_env['env']['observation_matrix'] and cfg_env['resnet_wrapper']:
                env = ResNetObservationWrapper(env)
            env = FlatObservationWrapper(env)
            #print(env.observation_space)
            if rank < 5 and video_enable:  # only add the RecordVideo wrapper for the first environment
                env = gym.wrappers.RecordVideo(env, name_prefix=f"{mode}_{rank}", video_folder=f"{output_dir}/videos_{mode}")  # record videos
            env = gym.wrappers.RecordEpisodeStatistics(env)  # rec0ord stats such as returns
            # if mode == 'rollout':
            #     env = RolloutInfoWrapper(env)  # record additional information in the rollout
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
    env = VecMonitor(env)
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

    #Rollout stats: {'n_traj': 2, 'return_min': 366.53949785232544, 'return_mean': 371.7009838819504, 'return_std': 5.161486029624939, 'return_max': 376.8624699115753, 'len_min': 148, 'len_mean': 148.0, 'len_std': 0.0, 'len_max': 148}

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
        tensorboard_log='./summary',
        verbose=1,
    )

    print("############# reward_net")
    reward_net = BasicRewardNet(
        observation_space=env.observation_space,
        action_space=env.action_space,
        normalize_input_layer=RunningNorm,
    )
    # Create a WandbOutputFormat instance
    wandb_format = imit_logger.WandbOutputFormat()


    custom_logger = imit_logger.configure(
        folder=output_dir,
        format_strs=["tensorboard", "stdout", "wandb"],
    )

    print("############# trainer")
    if cfg.irl_algo == "gail":
        trainer = GAIL(
            demonstrations=rollouts,
            demo_batch_size=cfg['irl_params']['demo_batch_size'],
            gen_replay_buffer_capacity=512,
            n_disc_updates_per_round=8,
            venv=env,
            gen_algo=learner,
            reward_net=reward_net,
            allow_variable_horizon=True,
            init_tensorboard = True,
            init_tensorboard_graph = True,
            log_dir = './summary',
            custom_logger=custom_logger,
        )
    elif cfg.irl_algo == "airl":
        trainer = AIRL(
            demonstrations=rollouts,
            demo_batch_size=cfg['irl_params']['demo_batch_size'],
            gen_replay_buffer_capacity=512,
            n_disc_updates_per_round=8,
            venv=env,
            gen_algo=learner,
            reward_net=reward_net,
            allow_variable_horizon=True,
            init_tensorboard = True,
            init_tensorboard_graph = True,
            log_dir = './summary',
            custom_logger=custom_logger,
        )

    checkpoint_interval = cfg['irl_params']['total_timesteps'] // cfg['irl_params']['num_checkpoints']
    def callback(round_num: int, /) -> None:
        if checkpoint_interval > 0 and  round_num % checkpoint_interval == 0:
            save(trainer, pathlib.Path(os.path.join(output_dir, f"checkpoints/checkpoint_{round_num:05d}")))


    print("############# Train")
    trainer.train(cfg['irl_params']['total_timesteps'], callback=callback)


    print("############# Save")
    save(trainer, pathlib.Path(os.path.join(output_dir, f"checkpoints/checkpointFinal")))


    cfg_env = cfg.copy()
    cfg_env['env']['copy_expert'] = False
    env_post = DummyVecEnv([make_env(mode='post', cfg_env=cfg_env, rank=i, seed=SEED,
                                 dict_scenarios=dict_scenarios, 
                                 list_scenarios=list_vessels_scenarios, video_enable = True) for i in range(cfg.num_envs)])

    env_post = VecMonitor(env_post)

    learner_rewards_after_training, _ = evaluate_policy(
        learner, env_post, cfg['irl_params']['num_eval_episodes'], return_episode_rewards=True)
    
    print("Rewards after training")
    print(learner_rewards_after_training)
    
    #learner_rewards_after_training = [-8.620693003417047, -8.620693003417047, 5.2321647564718194, 5.2321647564718194, 5.2321647564718194]
    wandb.log({"final_performance": np.mean(learner_rewards_after_training)})


    # At the end of your script, optionally
    wandb.finish()




if __name__ == "__main__":
    train()