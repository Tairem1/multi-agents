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
from train import wandb_init, set_seed

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
    set_seed(0)
    
    wandb.init(project="GCN-DRL", group="cartpole-tests", job_type="train")
    config = {'EPISODES_PER_EPOCH': 20,  
        'BATCH_SIZE': 32,
        'GAMMA': 0.99,
        'EPS_START': 0.9,
        'EPS_END': 0.05,
        'EPS_DECAY': 1000,
        'TOTAL_TIMESTEPS': 50_000,
        'LR': 1e-2,
        'NETWORK_UPDATE_FREQUENCY': 1,
        }
    wandb.config.update(config)
    
    from train import EpochCallback, CheckpointCallback, LossCallback
    
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
                 learning_rate=config['LR'],
                 batch_size=config['BATCH_SIZE'], 
                 start_learn = 50,
                 eps_start = config['EPS_START'],
                 eps_min=config['EPS_END'],
                 eps_decay=config['EPS_DECAY'],
                 replay_buffer_size=100_000, 
                 network_update_frequency=config['NETWORK_UPDATE_FREQUENCY'],
                 episodes_per_epoch=config['EPISODES_PER_EPOCH'])
    
    epoch_callback = EpochCallback(checkpoint_dir)
    checkpoint_callback = CheckpointCallback(checkpoint_dir, save_every=20_000)
    loss_callback = LossCallback()
    
    try:
        model.learn(total_timesteps=config['TOTAL_TIMESTEPS'], 
                    log=True, 
                    epoch_callback=epoch_callback,
                    checkpoint_callback=None,
                    loss_callback=loss_callback)
    finally:
        env.close()
    
    