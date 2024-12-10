#!/usr/bin/env python3
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib.image as mpimg
import numpy as np
import time
import pygame
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium.core import ActType

from utils import SDC_Wrapper, save_demonstrations, load_demonstrations


class ImageSlider:
    def __init__(self, observations):
        self.observations = observations
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.35)
        self.slider = None
        self.start_index = None
        self.end_index = None

    def update_image(self, val):
        index = int(self.slider.val)
        self.ax.imshow(self.observations[index])
        self.fig.canvas.draw_idle()

    def set_start_index(self, event):
        self.start_index = int(self.slider.val)
        print(f"Start index set to: {self.start_index}")

    def set_end_index(self, event):
        self.end_index = int(self.slider.val)
        print(f"End index set to: {self.end_index}")

    def show(self):
        self.ax.imshow(self.observations[0])
        ax_slider = plt.axes([0.25, 0.2, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        self.slider = Slider(ax_slider, 'Image', 0, len(self.observations)-1, valinit=0, valstep=1)
        self.slider.on_changed(self.update_image)

        ax_button_start = plt.axes([0.25, 0.05, 0.1, 0.075])
        btn_start = Button(ax_button_start, 'Set Start')
        btn_start.on_clicked(self.set_start_index)

        ax_button_end = plt.axes([0.75, 0.05, 0.1, 0.075])
        btn_end = Button(ax_button_end, 'Set End')
        btn_end.on_clicked(self.set_end_index)

        plt.show()


class ControlStatus:
    """
    Class to keep track of key presses while recording demonstrations.
    """
    def __init__(self):
        self.stop = False
        self.save = False
        self.quit = False

        self.steer = 0.0
        self.accelerate = 0.0
        self.brake = 0.0

    def update(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT: self.quit = True

            if event.type == pygame.KEYDOWN:
                self.key_press(event)

        keys = pygame.key.get_pressed()
        self.accelerate = 0.5 if keys[pygame.K_UP] else 0
        self.brake = 0.8 if keys[pygame.K_DOWN] else 0
        self.steer = 1 if keys[pygame.K_RIGHT] else (-1 if keys[pygame.K_LEFT] else 0)

    def key_press(self, event):
        if event.key == pygame.K_ESCAPE:    self.quit = True
        if event.key == pygame.K_SPACE:     self.stop = True
        if event.key == pygame.K_TAB:       self.save = True


def record_demonstrations(demonstrations_file):
    """
    Function to record own demonstrations by driving the car in the gym car-racing
    environment.
    demonstrations_file:  python string, the path to where the recorded demonstrations
                        are to be saved

    The controls are:
    arrow keys:         control the car; steer left, steer right, gas, brake
    ESC:                quit and close
    SPACE:              restart on a new track
    TAB:                save the current run
    """

    env = SDC_Wrapper(gym.make('CarRacing-v3', render_mode='human'), remove_score=True)

    status = ControlStatus()
    total_reward = 0.0

    while not status.quit:
        observations = []
        actions = []
        # get an observation from the environment
        observation = env.reset()

        while not status.stop and not status.save and not status.quit:
            status.update()

            # collect all observations and actions
            observations.append(observation.copy())
            actions.append(np.array([status.steer, status.accelerate,
                                    status.brake]))
            # submit the users' action to the environment and get the reward
            # for that step as well as the new observation (status)
            observation, reward, done, trunc = env.step([status.steer, 
                                                         status.accelerate,
                                                         status.brake])

            total_reward += reward
            time.sleep(0.01)

        if status.save:
            image_slider = ImageSlider(observations)
            image_slider.show()
            observations = observations[image_slider.start_index:image_slider.end_index+1]
            actions = actions[image_slider.start_index:image_slider.end_index+1]
            save_demonstrations(demonstrations_file, actions, observations)
            status.save = False

        status.stop = False

    env.close()


if __name__ == '__main__':
    if len(sys.argv) == 2:
        record_demonstrations(sys.argv[1])
    elif (len(sys.argv) > 2) and (sys.argv[1] in ['--view', '-v']):
        observations, _ = load_demonstrations(sys.argv[2])
        image_slider = ImageSlider(observations)
        image_slider.show()
    else:
        print("Usage: %s [--view] <data_file>. Exiting." % sys.argv[0])