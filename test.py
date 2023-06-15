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


agents = ['023']
models = [
    # "reward_best.pth", 
    "model_1900000.pth",
          ]
RENDER = False

for x in agents:
    checkpoint_dir = f"./checkpoint/reward_tuning/{x}/"
    models = glob.glob(os.path.join(checkpoint_dir, '*.pth'))
    best_model = {'model': None, 'metrics': {'collision': np.inf, 
                                             'goal_reached': 0, 
                                             'timeout':0, 
                                             'reward': 0.0}}
    for model in models:
        # model_pth = checkpoint_dir + "model_200000.pth"
        
        with open(checkpoint_dir+"args.pkl", 'rb') as f:
            args = pickle.load(f)
        
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
                    obs_type=args['POLICY_NETWORK'],
                    reward_configuration=args['REWARD'])
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
            eps = 0.0
            
            metrics = {'collision': 0, 'goal_reached': 0, 'timeout': 0, 'reward': 0.0}
            
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
                        # print(f"Total reward: {env.episode_reward}")
                        break
            metrics['reward'] = np.mean(rewards)        
                    
            if metrics['collision'] < best_model['metrics']['collision']:
                best_model['model'] = model
                best_model['metrics'] = metrics
        finally:
            env.close()
    print('*'*50)
    # print(f"TEST\tMean Reward: {best_model['reward']}")
    print(f"TEST\t {model} Metrics: {metrics}")
    print('*'*50)