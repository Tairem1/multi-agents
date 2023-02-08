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

def set_seed(seed):
    torch_geometric.seed_everything(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    set_seed(0)
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
    
    agent = GCNPolicy(num_node_features, action_size)
    model = DDQN(world, agent)
    model.learn(N_iter=100_000, log=True)
    