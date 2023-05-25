# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 08:01:05 2023

@author: AdminS
"""

import torch
import torch_geometric
import pickle

from scenes import Scene
from policy import init_agent

import numpy as np
import copy
import time

def prepare_state_for_nn(state):
    if isinstance(state, tuple):
        if len(state) == 2:
            s = (state[0], torch.tensor([[state[1]]]))
        elif len(state) == 3:
            s = (state[0], 
                 torch.tensor([[state[1]]], dtype=torch.float32),
                 torch.tensor([state[2]], dtype=torch.float32))
    else:
        raise Exception("Unsupported state type")
    return s

checkpoint_dir = "./checkpoint/05_19_000/"
model_pth = checkpoint_dir + "reward_best.pth"
# model_pth = checkpoint_dir + "model_1000000.pth"

with open(checkpoint_dir+"args.pkl", 'rb') as f:
    args = pickle.load(f)

dt = 0.2
env = Scene(dt, 
            width=120,
            height=120,
            ppm=5,
            render=True,
            discrete_actions=True,
            testing=True,
            seed=1234,
            obs_type=args['POLICY_NETWORK'],
            reward_configuration=args['REWARD'])
try:
    env.load_scene("scene01")
    
    agent = init_agent(Scene.ACTION_SIZE, 
                       Scene.OBS_SIZE, 
                       hidden_features=args['HIDDEN_FEATURES'],
                       obs_type=args['POLICY_NETWORK'])
    agent.eval()
    agent.load_state_dict(torch.load(model_pth))
    
    done = False
    N_EPISODES = 10
    rewards = []
    for _ in range(N_EPISODES):
        state, _ = env.reset()
        for _ in range(10_000):
            s = prepare_state_for_nn(state)
            action = int(torch.argmax(agent(s)))
            state, reward, terminated, truncated, info = env.step(action)
            time.sleep(dt/4.0)
            if terminated or truncated:
                rewards.append(env.episode_reward)
                print(f"Total reward: {env.episode_reward}")
                break
    print(f"TEST\tMean Reward: {np.mean(rewards)}")
finally:
    env.close()