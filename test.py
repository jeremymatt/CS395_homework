# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 21:54:24 2020

@author: jmatt
"""


import gym # openAi gym
from gym import envs

env = gym.make('FetchReach-v1')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()