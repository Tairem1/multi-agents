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


models = ['014', '013', '012']

for model in models:
    checkpoint_dir = f"./checkpoint/reward_tuning/{model}/"
    model_pth = checkpoint_dir + "reward_best.pth"
    # model_pth = checkpoint_dir + "model_200000.pth"
    
    with open(checkpoint_dir+"args.pkl", 'rb') as f:
        args = pickle.load(f)
    
    set_seed(0)
    RENDER = False
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
        agent.load_state_dict(torch.load(model_pth))
        
        seed = 0
        done = False
        N_EPISODES = 100
        rewards = []
        eps = 0.0
        
        metrics = {'collision': 0, 'goal_reached': 0, 'timeout': 0}
        
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
        print('*'*50)
        print(f"TEST\tMean Reward: {np.mean(rewards)}")
        print(f"TEST\t {model_pth} Metrics: {metrics}")
        print('*'*50)
    finally:
        env.close()