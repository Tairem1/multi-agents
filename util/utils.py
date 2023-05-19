# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 15:50:37 2023

@author: AdminS
"""
import torch
import numpy as np
import random
import torch_geometric
import wandb



def set_seed(seed):
    torch_geometric.seed_everything(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False   