# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 15:49:32 2023

@author: AdminS
"""
import wandb
import json
import torch
import os


class EpochCallback:
    def __init__(self, checkpoint_dir, wandb_log=True):
        self.metrics = {'epoch_reward': []}  # By default, an epoch is 100 episodes
        self.constant_metrics = {'epoch_number': -1, 
                                 'max_epoch_reward': -99999999}
        self.checkpoint_dir = checkpoint_dir
        self._wandb_log = wandb_log
        
        if self._wandb_log:
            wandb.run.define_metric("epoch_reward", step_metric="x_step")
    
    def __call__(self, epoch_reward, epoch_count, model):
        self.metrics['epoch_reward'].append(epoch_reward)
        
        if self._wandb_log:
            wandb.log({'epoch_reward': self.metrics['epoch_reward'][-1],
                       "x_step": epoch_count})
        
        if epoch_reward > self.constant_metrics['max_epoch_reward']:
            self.constant_metrics['epoch_number'] = epoch_count
            self.constant_metrics['max_epoch_reward'] = epoch_reward
            torch.save(model.state_dict(), 
                       os.path.join(self.checkpoint_dir, "reward_best.pth"))
            
    def __del__(self):
        with open(os.path.join(self.checkpoint_dir, "epoch_metrics.json"), "w") as f:
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
    def __init__(self, save_every=1, wandb_log=True):
        self.save_every = save_every
        self._wandb_log = wandb_log
        
        if self._wandb_log:
            wandb.run.define_metric("loss", step_metric="y_step")
        
    def __call__(self, loss, count):
        if count % self.save_every == 0:
            if self._wandb_log:
                wandb.log({'loss': loss.item(), "y_step": count})
            
class EvalCallback:
    def __init__(self, checkpoint_dir, wandb_log=True):
        self.metrics = {'eval_reward': [], "eval_std": []}  # By default, an epoch is 100 episodes
        self.constant_metrics = {'iteration': -1, 
                                 'max_eval_reward': -99999999}
        self.checkpoint_dir = checkpoint_dir
        self._wandb_log = wandb_log
        
        if self._wandb_log:
            wandb.run.define_metric("eval_reward", step_metric="z_step")
    
    def __call__(self, mean, std, count, model):
        self.metrics['eval_reward'].append(mean)
        self.metrics['eval_std'].append(std)
        
        if self._wandb_log:
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
    

        