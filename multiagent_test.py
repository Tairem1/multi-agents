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
from util.utils import set_seed

import numpy as np
import copy
import time
import glob
import os
from tqdm import tqdm

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

agent = 'vehicle'
pedestrian_level = "L0"
vehicle_level = "L1"
pedestrian_agent = "000" 
vehicle_agent = "001"
svo = 40

RENDER = True

base_dir =  f"./checkpoint/svo/{svo}/" if svo is not None else \
    "./checkpoint/svo/"
   
vehicle_path = os.path.join(base_dir, "vehicle", vehicle_level, vehicle_agent, "reward_best.pth")
pedestrian_path = os.path.join(base_dir, "pedestrian", pedestrian_level, pedestrian_agent, "reward_best.pth")
model = vehicle_path if agent.lower() == "vehicle" else pedestrian_path

# models = glob.glob(os.path.join(checkpoint_dir, '*.pth'))
# best_model = {'model': None, 'metrics': {'collision': np.inf, 
#                                          'goal_reached': 0, 
#                                          'timeout':0, 
#                                          'reward': 0.0}}

if agent.lower() == "vehicle":
    pkl_file = os.path.join(base_dir, agent, vehicle_level, vehicle_agent, "args.pkl")
else:
    pkl_file = os.path.join(base_dir, agent, pedestrian_level, pedestrian_agent, "args.pkl")


with open(pkl_file, 'rb') as f:
    args = pickle.load(f)
    print(args)

print(f"Vehicle path: {vehicle_path}")
print(f"Pedestrian path: {pedestrian_path}")

set_seed(0)
env = Scene(args['dt'], 
            width=120,
            height=120,
            ppm=5,
            render=RENDER,
            window_name="Testing",
            discrete_actions=True,
            testing=True,
            seed=0,
            reward_configuration=args['REWARD'],
            agent=agent,
            
            vehicle_level=vehicle_level,
            pedestrian_level=pedestrian_level,
            hidden_features=args['HIDDEN_FEATURES'],
            obs_type=args['POLICY_NETWORK'],
            gcn_conv_layer=args['GCN_CONV_LAYER'],
            n_conv_layers=args['N_CONV_LAYERS'],
            path_to_vehicle_agent=vehicle_path,
            path_to_pedestrian_agent=pedestrian_path,
            svo=svo)
try:
    env.load_scene("scene01")
    
    agent = init_agent(Scene.ACTION_SIZE, 
                       Scene.OBS_SIZE, 
                       hidden_features=args['HIDDEN_FEATURES'],
                       obs_type=args['POLICY_NETWORK'],
                       gcn_conv_layer=args['GCN_CONV_LAYER'],
                       n_conv_layers=args['N_CONV_LAYERS']
                       )
    agent.eval()
    agent.load_state_dict(torch.load(model))
    
    seed = 0
    done = False
    N_EPISODES = 100
    rewards = []
    ep_lengths = []
    eps = 0.0
    
    metrics = {'collision': 0, 'goal_reached': 0, 'timeout': 0, 'reward': 0.0, 'avg ep len': 0.0}
    
    for _ in tqdm(range(N_EPISODES)):
        seed += 1
        env.reset_rng(seed)
        state, data = env.reset()
        for _ in range(10_000):
            s = prepare_state_for_nn(state)
            # print(agent(s))
            action = int(torch.argmax(agent(s)))
            
            if np.random.uniform() < eps:
                action = np.random.randint(0, 3)
            
            state, reward, terminated, truncated, info = env.step(action)
            # time.sleep(dt/4.0)
            if RENDER:
                time.sleep(args['dt']/4.0)
            
            if terminated or truncated:
                metrics[info['end_reason']] += 1
                rewards.append(env.episode_reward)
                ep_lengths.append(env.t)
                # print(f"Total reward: {env.episode_reward}")
                break
    metrics['reward'] = np.mean(rewards)
    metrics['avg ep len'] = np.mean(ep_lengths)        
finally:
    env.close()
    
print('*'*50)
# print(f"TEST\tMean Reward: {best_model['reward']}")
print(f"TEST\t {model} Metrics: {metrics}")
print('*'*50)