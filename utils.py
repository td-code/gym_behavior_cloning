import numpy as np
import time
import pygame

import gymnasium as gym
from gymnasium.core import ActType


class SDC_Wrapper(gym.Wrapper):
    """
    Our main interface to access the simulation enviorment.
    This provides a reset and step function that is compatible with the gym interface.
    """
    def __init__(self, env, remove_score=True):
        super().__init__(env)

        self.remove_score = remove_score

    def reset(self, **kwargs):
        observation, _ = super().reset(**kwargs)

        if self.remove_score:
            observation[84:, :11, :] = 0

        return observation

    def step(self, action: ActType):
        observation, reward, done, truncated, _ = super().step(np.array(action))
        reward_clipped = np.clip(reward, -0.1, 1e8)

        if self.remove_score:
            observation[84:, :11, :] = 0

        return observation, reward_clipped, done, truncated
    
    
def load_demonstrations(data_file):
    """
    The data gets loaded and stored it in two lists: observations and actions.
                    N = number of (observation, action) - pairs
    data_file:      python string, the recorded data file as .npz (may also be a list of files)
    return:
    observations:   python list of N numpy.ndarrays of size (96, 96, 3)
    actions:        python list of N numpy.ndarrays of size 3
    """
    if not isinstance(data_file, list):
        data_file = [data_file]

    observations = []
    actions = []    
    for f in data_file:
        data = np.load(f)
        observations.extend(list(data['observations']))
        actions.extend(list(data['actions']))

    return observations, actions


def save_demonstrations(data_file, actions, observations):
    """
    Save the lists actions and observations in numpy .npz file that can be read
    by the function load_demonstrations.
                    N = number of (observation, action) - pairs
    data_folder:    python string, the recorded data file as .npz
    observations:   python list of N numpy.ndarrays of size (96, 96, 3)
    actions:        python list of N numpy.ndarrays of size 3
    """
    np.savez_compressed(data_file, observations=observations, actions=actions)