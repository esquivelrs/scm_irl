import gymnasium as gym
from gymnasium.spaces import Box, Dict
import torch
import numpy as np
from torchvision.models import resnet18
from torchvision import models, transforms
import math


class ResNetObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.resnet18 = models.resnet18(pretrained=True)
        # Modify the ResNet model to remove the fully connected layer
        self.resnet18 = torch.nn.Sequential(*(list(self.resnet18.children())[:-2]))
        self.resnet18 = self.resnet18.double()  # Set the model to double precision
        self.resnet18.eval()  # Set the model to evaluation mode
        
        # Calculate the output size of the second-last layer for the given input dimensions
        # This is a placeholder; you'll need to empirically determine this or calculate based on ResNet architecture
        
        # Read the H, W, and C dimensions from the original observation space
        H, W, C = self.observation_space['observation_matrix'].shape
        
        # Calculate the output size of the second-last layer for the given input dimensions
        output_size = math.ceil(H / 32) * math.ceil(W / 32) * 512
        
        # Update the observation space to include the new part
        self.observation_space = Dict({
            'agent_state': env.observation_space['agent_state'],
            'observation_matrix': Box(-np.inf, np.inf, (1,), dtype=np.float32),
            'vessel_params': env.observation_space['vessel_params']
        })

    def observation(self, observation):
        # Process the observation_matrix through ResNet
        obs_matrix = observation['observation_matrix']
        # Convert observation matrix to tensor, normalize, and add batch dimension
        obs_matrix = transforms.functional.to_tensor(obs_matrix).unsqueeze(0).double()
        with torch.no_grad():
            resnet_output = self.resnet18(obs_matrix)
        # Flatten the output
        resnet_features = resnet_output.flatten(start_dim=1)
        
        # Update the observation dictionary
        resnet_fm = resnet_features.numpy()
        observation['observation_matrix'] = resnet_fm[0][0]
        return observation



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