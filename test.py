# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 08:01:05 2023

@author: AdminS
"""

import torch
import torch_geometric
import pickle

from scenes import Scene
from policy import GCNPolicy

import numpy as np
import copy

def eps_greedy_action_selection(eps, model, state, device='cpu'):
    with torch.no_grad():
        p = np.random.rand()
        if p < eps:
            # Random action
            action = np.random.randint(model.num_actions)
        else:
            # Action selection based on Q function
            if isinstance(state, np.ndarray):
                s = torch.tensor(state, device=device)
            elif isinstance(state, torch_geometric.data.Data):
                s = copy.deepcopy(state).to(device)
            else:
                raise Exception(f"Unsupported state type: {type(state)}")
            action = int(torch.argmax(model(s)))
    return action

checkpoint_dir = "./checkpoint/dqn_gcn_012/"
# model_pth = checkpoint_dir + "reward_best.pth"
model_pth = checkpoint_dir + "model_990000.pth"

with open(checkpoint_dir+"args.pkl", 'rb') as f:
    args = pickle.load(f)

dt = 0.2
env = Scene(dt, 
            width=120,
            height=120,
            ppm=5,
            render=True,
            discrete_actions=True)
env.load_scene("scene01")
agent = GCNPolicy(Scene.OBS_SIZE, 
                  Scene.ACTION_SIZE,
                  hidden_features=args['HIDDEN_FEATURES'])
agent.load_state_dict(torch.load(model_pth))

done = False
N_EPISODES = 1
for _ in range(N_EPISODES):
    state, _ = env.reset()
    for _ in range(10_000):
        action = 0#eps_greedy_action_selection(0.05, agent, state)
        state, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print(f"Total reward: {env.episode_reward}")
            break
env.close()