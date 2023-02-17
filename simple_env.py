# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 08:42:32 2023

@author: AdminS
"""
import torch
from drl.ddqn import DDQN
import gymnasium as gym
import numpy as np
import os
import wandb

class MLPPolicy(torch.nn.Module):
    def __init__(self, state_dim, num_actions, hidden_dim=5):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, num_actions)
            )
        self.num_actions = num_actions  
        
        
    def forward(self, x):
        return self.model(x)
    
    
    
    
if __name__ == "__main__":
    BATCH_SIZE = 128
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 1000
    LR = 1e-2
    
    from train import EpochCallback, CheckpointCallback
    
    checkpoint_dir = "./checkpoint/cartpole"
    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    
    env = gym.make('CartPole-v1', render_mode="human")
    # env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = MLPPolicy(state_size, action_size).to(device)
    model = DDQN(env, agent, device, 
                 learning_rate=LR,
                 batch_size=BATCH_SIZE, 
                 start_learn = 50,
                 eps_start = EPS_START,
                 eps_min=EPS_END,
                 eps_decay=EPS_DECAY,
                 replay_buffer_size=100_000, 
                 network_update_frequency=50)
    
    epoch_callback = EpochCallback(checkpoint_dir)
    checkpoint_callback = CheckpointCallback(checkpoint_dir, save_every=20_000)
    try:
        model.learn(total_timesteps=10_000, 
                    log=True, 
                    epoch_callback=None,
                    checkpoint_callback=None)
    finally:
        env.close()
    
    