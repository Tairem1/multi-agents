# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 17:22:55 2023

@author: lucac
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import aggr

class GCNPolicy(torch.nn.Module):
    def __init__(self, num_node_features, num_actions, hidden_features=16):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_features)
        self.ReLU1 = torch.nn.ReLU()
        self.conv2 = GCNConv(hidden_features, hidden_features)
        self.ReLU2 = torch.nn.ReLU()
        # self.global_pooling = aggr.MeanAggregation()
        self.global_pooling = aggr.MaxAggregation()
        self.classifier = torch.nn.Linear(hidden_features, num_actions)
        self._num_actions = num_actions
        
    @property
    def num_actions(self):
        return self._num_actions
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.global_pooling(x)
        x = self.classifier(x)
        return x.squeeze()
                
        
if __name__ == "__main__":
    import numpy as np
    import random
    from torch_geometric.data import Data
    import torch_geometric
    
    def set_seed(seed):
        torch_geometric.seed_everything(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    N_FEATURES = 4
    N_NODES = 3
    N_ACTIONS = 5
    
    set_seed(2)
    model = GCNPolicy(N_FEATURES, N_ACTIONS)
    model.eval()
    nodes = torch.rand((N_NODES, N_FEATURES))
    edge_index = torch.tensor([[0,1], [1,0], [0,2]])
    nodes1 = nodes[[0,2,1], :]
    edge_index1 = torch.tensor([[0, 2], [2,0], [0,1]])
    
    obs = Data(x=nodes, edge_index=edge_index.t().contiguous())
    obs1 = Data(x=nodes1, edge_index=edge_index1.t().contiguous())
    
    print(model(obs))
    print(model(obs1))
    







    
