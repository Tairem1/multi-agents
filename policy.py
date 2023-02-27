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
    def __init__(self, n_features, n_actions, hidden_features=16):
        super().__init__()
        self._n_features = n_features
        self._n_actions = n_actions
        
        self.conv1 = GCNConv(n_features, hidden_features)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = GCNConv(hidden_features, hidden_features)
        self.relu2 = torch.nn.ReLU()
        
        self.global_pooling = aggr.MaxAggregation()
        self.linear1 = torch.nn.Linear(hidden_features, n_actions)
        
    def forward(self, batch):
        x = self.conv1(batch.x, batch.edge_index, batch.edge_weight)
        x = self.relu1(x)
        x = self.conv2(x, batch.edge_index, batch.edge_weight)
        x = self.relu2(x)
        x = self.global_pooling(x, batch.batch)
        x = self.linear1(x)
        return x.squeeze()
    
    @property
    def num_actions(self):
        return self._n_actions
    
    @property
    def num_node_features(self):
        return self._n_features
        
    
class GCNPolicyTuple(torch.nn.Module):
    def __init__(self, n_features, n_actions, hidden_features=16):
        super().__init__()
        self._n_features = n_features
        self._n_actions = n_actions
        
        self.conv1 = GCNConv(n_features, hidden_features)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = GCNConv(hidden_features, hidden_features)
        self.relu2 = torch.nn.ReLU()
        
        self.global_pooling = aggr.MaxAggregation()
        
        speed_dim = 4
        self.speed_encoder = torch.nn.Linear(1, speed_dim)
        self.linear = torch.nn.Linear(hidden_features + speed_dim, n_actions)
        
    def forward(self, batch):
        if isinstance(batch, tuple):
            graph_batch, speed_batch = batch
        else:
            raise(Exception("Data must be of batch type"))
            
        x = self.conv1(graph_batch.x, graph_batch.edge_index, graph_batch.edge_weight)
        x = self.relu1(x)
        x = self.conv2(x, graph_batch.edge_index, graph_batch.edge_weight)
        x = self.relu2(x)
        x = self.global_pooling(x, graph_batch.batch)
        
        v = self.speed_encoder(speed_batch)
        h = torch.cat((x,v), dim=1)
        h = self.linear(h)
        
        return h.squeeze()
    
    @property
    def num_actions(self):
        return self._n_actions
    
    @property
    def num_node_features(self):
        return self._n_features
            
        
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
    







    
