# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 17:10:09 2023

@author: lucac
"""

from scenes import Scene
from drl.ddqn import DDQN
from policy import GCNPolicy
import torch
import numpy as np
import random
import torch_geometric

import os
import pickle
import json
import wandb
import argparse
from util.utils import set_seed
from util.callbacks import EpochCallback,CheckpointCallback,EvalCallback,LossCallback
    
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', type=str, default="dqn_gcn",
                        help="Model name.")
    parser.add_argument('--group_name', type=str, default=None,
                        help="Specify an experiment group name to group together multiple runs")
    parser.add_argument('--log', action="store_true", default=False) 
    
    parser.add_argument('--seed', type=int, default=0,
                        help="Seed for random initialisation.")             
    parser.add_argument('--wandb', action="store_true", default=False)   
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=16, 
                        help="Minibatch Size.")
    parser.add_argument('--gamma', type=float, default=0.99,
                        help="discount factor.")
    parser.add_argument('--learning_rate', type=float, default=5.0e-04,
                        help="learning rate for optimizer.")
    parser.add_argument('--total_timesteps', type=int, default=100_000,
                        help="The total of number of samples to train on.")
    args = parser.parse_args()
    return args


def wandb_init():
    wandb.init(project="GCN-DRL", group="Intersection", job_type="train")
    wandb.config.update(args)
    
def create_checkpoint_directory():
    if not os.path.isdir("./checkpoint/"):
        os.mkdir("./checkpoint/")
    for i in range(1000):
        checkpoint_dir = f"./checkpoint/{args.tag}_{i:03d}/"
        if not os.path.isdir(checkpoint_dir):
            os.mkdir(checkpoint_dir)
            return checkpoint_dir
    raise Exception("Unable to create checkpoint directory.")
    
def save_args(checkpoint_dir, args): 
    with open(checkpoint_dir+'args.pkl', 'wb') as fp:
        pickle.dump(args, fp)
    
        

if __name__ == "__main__":
    args = parse()
    
    config = {'EPISODES_PER_EPOCH': 20,  
        'BATCH_SIZE': 8,
        'GAMMA': 0.99,
        'EPS_START': 1.0,
        'EPS_END': 0.05,
        'EPS_DECAY': 50_000,
        'TOTAL_TIMESTEPS': 1_000_000,
        'REPLAY_BUFFER_SIZE': 10_000,
        'LR': 1e-4,
        'NETWORK_UPDATE_FREQUENCY': 50,
        'SEED': args.seed,
        }
    
    if args.wandb:
        wandb_init()
        wandb.config.update(config)
    
    set_seed(config['SEED'])
    num_node_features = 4
    action_size = 3
        
    dt = 0.2 # time steps in terms of seconds. In other words, 1/dt is the FPS.
    
    # The world is 120 meters by 120 meters. ppm is the pixels per meter.
    world = Scene(dt, 
                  width = 120, 
                  height = 120, 
                  ppm = 5, 
                  render=True,
                  discrete_actions=True)
    
    test_world = Scene(dt, 
                  width = 120, 
                  height = 120, 
                  ppm = 5, 
                  render=False,
                  discrete_actions=True)
    
    world.load_scene("scene01")
    test_world.load_scene("scene01")
    
    print('*'*30)
    print("Training initiating....")
    print(config)
    print()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = GCNPolicy(num_node_features, action_size).to(device)
    model = DDQN(world, 
                  agent, 
                  test_env=None, 
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
     
    
    # wandb.watch(agent, log="all", log_freq=100)

    checkpoint_dir = create_checkpoint_directory()
    print(f'Saving data to: {checkpoint_dir}')
    save_args(checkpoint_dir, config)
    
    checkpoint_callback = CheckpointCallback(checkpoint_dir, save_every=5_000)
    try:
        model.learn(total_timesteps=config['TOTAL_TIMESTEPS'], 
                    log=args.log, 
                    epoch_callback=EpochCallback(checkpoint_dir,wandb_log=args.wandb),
                    checkpoint_callback=checkpoint_callback,
                    loss_callback=LossCallback(wandb_log=args.wandb),
                    eval_callback=EvalCallback(checkpoint_dir, wandb_log=args.wandb))
    finally:
        world.close()
        wandb.finish()
    