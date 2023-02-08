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
import wandb
import argparse

def set_seed(seed):
    torch_geometric.seed_everything(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class EpochCallback:
    def __init__(self, checkpoint_dir):
        self.metrics = {'epoch_reward': []}  # By default, an epoch is 100 episodes
        self.constant_metrics = {'epoch_number': -1, 
                                 'max_epoch_reward': -99999999}
        self.checkpoint_dir = checkpoint_dir
    
    def __call__(self, epoch_reward, epoch_count, model):
        self.metrics['epoch_reward'].append(epoch_reward)
        wandb.log({'epoch_reward': self.metrics['epoch_reward'][-1]})
        
        if epoch_reward > self.constant_metrics['min_epoch_reward']:
            self.constant_metrics['epoch_number'] = epoch_count
            self.constant_metrics['max_epoch_reward'] = epoch_reward
            torch.save(model.state_dict(), checkpoint_dir + "reward_best.pth")
            
    
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', type=str, default="dqn_gcn",
                        help="Model name.")
    parser.add_argument('--group_name', type=str, default=None,
                        help="Specify an experiment group name to group together multiple runs")
    parser.add_argument('--log', action="store_true", default=False,
                        help="Log training curves") 
    parser.add_argument('--seed', type=int, default=0,
                        help="Seed for random initialisation.")             
    
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
    wandb.init(project="GCN-DRL", group=args.group_name, job_type="train")
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
    wandb_init()
    
    set_seed(args.seed)
    num_node_features = 4
    action_size = 3
        
    dt = 0.1 # time steps in terms of seconds. In other words, 1/dt is the FPS.
    
    # The world is 120 meters by 120 meters. ppm is the pixels per meter.
    world = Scene(dt, 
                  width = 120, 
                  height = 120, 
                  ppm = 5, 
                  render=True,
                  discrete_actions=True)
    world.load_scene("scene01")
    
    print('*'*30)
    print("Training initiating....")
    print(args)
    
    
    agent = GCNPolicy(num_node_features, action_size)
    model = DDQN(world, 
                 agent, 
                 learning_rate=args.learning_rate,
                 gamma=args.gamma,
                 batch_size=args.batch_size,
                 episodes_per_epoch=100)
    
    wandb.watch(agent, log="all", log_freq=100)

    checkpoint_dir = create_checkpoint_directory()
    print(f'Saving data to: {checkpoint_dir}')
    
    
    save_args(checkpoint_dir, args)
    
    epoch_callback = EpochCallback(checkpoint_dir)
    model.learn(total_timesteps=args.total_timesteps, 
                log=args.log, 
                epoch_callback=epoch_callback)
    
    
    with open(checkpoint_dir+'metrics.pkl', 'wb') as fp:
        pickle.dump(epoch_callback.metrics, fp)
        
    world.close()
    