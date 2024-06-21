import gymnasium as gym
from gymnasium.spaces import Box, Dict
import torch
import numpy as np
from torchvision.models import resnet18
from torchvision.transforms import ToTensor, Normalize, Compose

class NetObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.resnet18 = resnet18(pretrained=True)
        self.resnet18.eval()  # Set the model to evaluation mode
        self.transforms = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def observation(self, obs):
        # Transform the observation here
        transformed_obs = self.transform_observation(obs)
        return transformed_obs

    def transform_observation(self, obs):
        # Apply the ResNet18 model to the observation_matrix
        observation_matrix = obs['observation_matrix']
        observation_matrix = self.transforms(observation_matrix)
        with torch.no_grad():
            observation_matrix = self.resnet18(observation_matrix.unsqueeze(0))

        transformed_obs = {
            'observation_matrix': observation_matrix,
            'agent_state': obs['agent_state'],
            'expert_state': obs['expert_state'],
            'target': obs['target']
        }
        return transformed_obs



class FlatObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.original_observation_space = env.observation_space
        
        # Calculate the total flat size
        total_size = sum(np.prod(space.shape) for space in env.observation_space.spaces.values())
        
        # Define the new flat observation space
        self.observation_space = Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(total_size,), 
            dtype=np.float32
        )

    def observation(self, observation):
        # Flatten each part of the observation and concatenate them
        flat_obs = np.concatenate([
            np.array(observation[key]).flatten() for key in self.original_observation_space.spaces
        ])
        return flat_obs