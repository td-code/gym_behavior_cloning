import numpy as np
import time
import time
import cv2
import os

from IPython.core.display import Video, display
from moviepy.editor import *

import gymnasium as gym
from gymnasium.core import ActType

import warnings
warnings.filterwarnings("ignore", message='Warning: in file')


class SDC_Wrapper(gym.Wrapper):
    """
    Our main interface to access the simulation enviorment.
    This provides a reset and step function that is compatible with the gym interface.
    """
    def __init__(self, env, remove_score=True):
        super().__init__(env)

        self.remove_score = remove_score

    def reset(self, **kwargs):
        observation, info = super().reset(**kwargs)

        if self.remove_score:
            observation[84:, :11, :] = 0

        return observation, info

    def step(self, action: ActType):
        observation, reward, done, truncated, info = super().step(np.array(action))
        reward_clipped = np.clip(reward, -0.1, 1e8)

        if self.remove_score:
            observation[84:, :11, :] = 0

        return observation, reward_clipped, done, truncated, info
    
    
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
    
    
class RenderFrame(SDC_Wrapper):
    """
    This class is used to play the simulation frames in a Jupyter notebook. 
    Code is taken from the render_frame library.
    """
    def __init__(self, env, directory, auto_release=True, size=None, fps=None, rgb=True):
        super().__init__(env)
        self.directory = directory
        self.auto_release = auto_release
        self.active = True
        self.rgb = rgb

        if env.render_mode != "rgb_array":
            raise Exception("RenderFrame requires environment render mode configured to rgb_array")

        os.makedirs(self.directory, exist_ok = True)

        if size is None:
            self.env.reset()
            self.size = self.env.render().shape[:2][::-1]
        else:
            self.size = size

        if fps is None:
            if 'video.frames_per_second' in self.env.metadata:
                self.fps = self.env.metadata['video.frames_per_second']
            else:
                self.fps = 30
        else:
            self.fps = fps

    def pause(self):
        self.active = False

    def resume(self):
        self.active = True

    def _start(self):
        if self.active:
            self.cliptime = time.time()
            self.path = f'{self.directory}/{self.cliptime}.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self._writer = cv2.VideoWriter(self.path, fourcc, self.fps, self.size)

    def _write(self):
        frame = self.env.render()
        if self.active:
            if self.rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._writer.write(frame)

    def release(self):
        if self.active:
            self._writer.release()

    def reset(self, *args, **kwargs):
        observation, info = self.env.reset(*args, **kwargs)
        self._start()
        self._write()
        return observation, info

    def step(self, *args, **kwargs):
        observation, reward, terminated, truncated, info = self.env.step(*args, **kwargs)
        self._write()

        if self.auto_release and (terminated or truncated):
            self.release()

        return observation, reward, terminated, truncated, info

    def play(self):
        start = time.time()
        filename = r'temp-{start}.mp4'
        clip = VideoFileClip(self.path, audio=False)
        clip.write_videofile(filename, verbose = False, logger = None)
        display(Video(filename, embed = True))
        os.remove(filename)
