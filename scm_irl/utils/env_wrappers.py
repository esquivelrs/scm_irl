import gymnasium as gym
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
        # observation space
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.observation_space['agent_state'].shape[0] +
                   self.observation_space['expert_state'].shape[0] +
                   self.observation_space['target'].shape[0],),
            dtype=np.float32
        )

    def observation(self, obs):
        # Transform the observation here
        self.obs = self.transform_observation(obs)
        return self.obs

    def transform_observation(self, obs):
        # Define your transformation here
        # This is just a placeholder example
        transformed_obs = {
            'agent_state': obs['agent_state'],
            'expert_state': obs['expert_state'],
            'target': obs['target']
        }
        # Convert all values to numpy arrays
        transformed_obs_values = [np.array(v) for v in transformed_obs.values()]

        # Concatenate all values into a single vector
        concatenated_vector = np.concatenate(transformed_obs_values)
        print('concatenated_vector.shape')
        print(concatenated_vector.shape)

        return concatenated_vector