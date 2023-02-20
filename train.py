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
        wandb.run.define_metric("epoch_reward", step_metric="x_step")
    
    def __call__(self, epoch_reward, epoch_count, model):
        self.metrics['epoch_reward'].append(epoch_reward)
        wandb.log({'epoch_reward': self.metrics['epoch_reward'][-1],
                   "x_step": epoch_count})
        
        if epoch_reward > self.constant_metrics['max_epoch_reward']:
            self.constant_metrics['epoch_number'] = epoch_count
            self.constant_metrics['max_epoch_reward'] = epoch_reward
            torch.save(model.state_dict(), 
                       os.path.join(self.checkpoint_dir, "reward_best.pth"))
            
    def __del__(self):
        with open(os.path.join(self.checkpoint_dir, "constant_metrics.json"), "w") as f:
            json.dump(self.constant_metrics, f, indent=2)
            
            
class CheckpointCallback:
    def __init__(self, checkpoint_dir, save_every):
        self.checkpoint_dir = checkpoint_dir
        self._n_calls = 0
        self._save_every = save_every
        assert(self._save_every > 100)
        
    def __call__(self, model):
        self._n_calls += 1
        if (self._n_calls % self._save_every) == 0:
            torch.save(model.state_dict(), 
                       os.path.join(self.checkpoint_dir, self.next_model_name))   
    @property
    def next_model_name(self):
        return f"model_{self._n_calls}.pth"  
    
    
class LossCallback:
    def __init__(self, save_every=1):
        self.save_every = save_every
        wandb.run.define_metric("loss", step_metric="y_step")
        
    def __call__(self, loss, count):
        if count % self.save_every == 0:
            wandb.log({'loss': loss.item(), "y_step": count})
            
class EvalCallback:
    def __init__(self, checkpoint_dir):
        self.metrics = {'eval_reward': [], "eval_std": []}  # By default, an epoch is 100 episodes
        self.constant_metrics = {'iteration': -1, 
                                 'max_eval_reward': -99999999}
        self.checkpoint_dir = checkpoint_dir
        wandb.run.define_metric("eval_reward", step_metric="z_step")
    
    def __call__(self, mean, std, count, model):
        self.metrics['eval_reward'].append(mean)
        self.metrics['eval_std'].append(std)
        wandb.log({'eval_reward': self.metrics['eval_reward'][-1],
                   "z_step": count})
        
        if mean > self.constant_metrics['max_eval_reward']:
            self.constant_metrics['iteration'] = count
            self.constant_metrics['max_eval_reward'] = mean
            torch.save(model.state_dict(), 
                       os.path.join(self.checkpoint_dir, "eval_best.pth"))
            
    def __del__(self):
        with open(os.path.join(self.checkpoint_dir, "eval_metrics.json"), "w") as f:
            json.dump(self.constant_metrics, f, indent=2)
    

        
    
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
    world.load_scene("scene01")
    
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
    
    epoch_callback = EpochCallback(checkpoint_dir)
    checkpoint_callback = CheckpointCallback(checkpoint_dir, save_every=20_000)
    model.learn(total_timesteps=config['TOTAL_TIMESTEPS'], 
                log=False, 
                epoch_callback=epoch_callback,
                checkpoint_callback=checkpoint_callback)
    
    with open(checkpoint_dir+'metrics.pkl', 'wb') as fp:
        pickle.dump(epoch_callback.metrics, fp)
        
    world.close()
    