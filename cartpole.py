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
import argparse

class MLPPolicy(torch.nn.Module):
    def __init__(self, state_dim, num_actions, hidden_dim=32):
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
    
    def act(self, x):
        return int(torch.argmax(self.forward(x)))
    
    
def create_checkpoint_directory():
    if not os.path.isdir("./checkpoint/"):
        os.mkdir("./checkpoint/")
    for i in range(1000):
        checkpoint_dir = f"./checkpoint/{args.tag}_{i:03d}/"
        if not os.path.isdir(checkpoint_dir):
            os.mkdir(checkpoint_dir)
            return checkpoint_dir
    raise Exception("Unable to create checkpoint directory.")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', type=str, default="cartpole",
                        help="Model name.")
    parser.add_argument('--seed', type=int, default=0,
                        help="Seed for random initialisation.")     
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    wandb.init(project="GCN-DRL", group="cartpole-tests", job_type="train")
    config = {'EPISODES_PER_EPOCH': 20,  
        'BATCH_SIZE': 32,
        'GAMMA': 0.99,
        'EPS_START': 1.0,
        'EPS_END': 0.01,
        'EPS_DECAY': 30_000,
        'TOTAL_TIMESTEPS': 50_000,
        'REPLAY_BUFFER_SIZE': 10_000,
        'LR': 1e-3,
        'NETWORK_UPDATE_FREQUENCY': 50,
        'SEED': args.seed,
        }
    wandb.config.update(config)
    
    from train import EpochCallback, CheckpointCallback, LossCallback, EvalCallback
    
    checkpoint_dir = create_checkpoint_directory()
    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    
    # env = gym.make('CartPole-v1', render_mode="human")
    env = gym.make('CartPole-v1', render_mode="human")
    test_env = gym.make('CartPole-v1')
    
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = MLPPolicy(state_size, action_size).to(device)
    model = DDQN(env, 
                 agent, 
                 test_env=test_env, 
                 device=device,
                 learning_rate=config['LR'],
                 batch_size=config['BATCH_SIZE'], 
                 start_learn = 1_000,
                 eps_start = config['EPS_START'],
                 eps_min=config['EPS_END'],
                 eps_decay=config['EPS_DECAY'],
                 replay_buffer_size=config['REPLAY_BUFFER_SIZE'], 
                 network_update_frequency=config['NETWORK_UPDATE_FREQUENCY'],
                 episodes_per_epoch=config['EPISODES_PER_EPOCH'])
    
    checkpoint_callback = CheckpointCallback(checkpoint_dir, save_every=5_000)
    try:
        model.learn(total_timesteps=config['TOTAL_TIMESTEPS'], 
                    log=False, 
                    epoch_callback=EpochCallback(checkpoint_dir),
                    checkpoint_callback=checkpoint_callback,
                    loss_callback=LossCallback(),
                    eval_callback=EvalCallback(checkpoint_dir))
    finally:
        env.close()
    
    