# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 17:22:55 2023

@author: lucac
"""

import torch
from torch_geometric.nn import GCNConv
from torch_geometric.nn import aggr
from torch_geometric.nn import BatchNorm as gBatchNorm
import torch.nn as nn
import torch.nn.functional as F
    
    
class GCNPolicy(nn.Module):
    def __init__(self, n_features, n_actions, hidden_features=16):
        super().__init__()
        self._n_features = n_features
        self._n_actions = n_actions
        
        self.conv1 = GCNConv(n_features, hidden_features)
        self.relu1 = nn.ReLU()
        self.conv2 = GCNConv(hidden_features, hidden_features)
        self.relu2 = nn.ReLU()
        
        self.global_pooling = aggr.MaxAggregation()
        self.linear1 = nn.Linear(hidden_features, n_actions)
        
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
        
    
class GCNPolicySpeed(nn.Module):
    def __init__(self, n_features, n_actions, hidden_features=16):
        super().__init__()
        self._n_features = n_features
        self._n_actions = n_actions
        
        self.conv1 = GCNConv(n_features, hidden_features)
        self.relu1 = nn.ReLU()
        self.conv2 = GCNConv(hidden_features, hidden_features)
        self.relu2 = nn.ReLU()
        
        self.global_pooling = aggr.MaxAggregation()
        
        speed_dim = 4
        self.speed_encoder = nn.Linear(1, speed_dim)
        self.linear1 = nn.Linear(hidden_features + speed_dim, 16)
        self.relu3 = nn.ReLU()
        self.linear2 = nn.Linear(16, n_actions)
        
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
        h = self.linear1(h)
        h = self.relu3(h)
        h = self.linear2(h)
        
        return h.squeeze()
    
    @property
    def num_actions(self):
        return self._n_actions
    
    @property
    def num_node_features(self):
        return self._n_features
    
    
class GCNPolicySpeedRoute(nn.Module):
    def __init__(self, n_features, n_actions, hidden_features=16):
        super().__init__()
        INPUT_ROUTE_LEN = 10
        ENCODED_SPEED_DIM = 4
        ENCODED_ROUTE_DIM = 4

        self._n_features = n_features
        self._n_actions = n_actions
        
        self.conv1 = GCNConv(n_features, hidden_features)
        self.batch_norm1 = gBatchNorm(hidden_features)
        self.relu1 = nn.ReLU()
        self.conv2 = GCNConv(hidden_features, hidden_features)
        self.batch_norm2 = gBatchNorm(hidden_features)
        self.relu2 = nn.ReLU()
        
        self.global_pooling = aggr.MaxAggregation()
        
        self.speed_encoder = nn.Sequential(
            nn.Linear(1, ENCODED_SPEED_DIM),
            nn.BatchNorm1d(ENCODED_SPEED_DIM),
            nn.ReLU())
        
        self.route_encoder = nn.Sequential(
            nn.Conv1d(2, 1, kernel_size=3, padding=1),
            nn.BatchNorm1d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(INPUT_ROUTE_LEN, ENCODED_ROUTE_DIM))
        
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_features + \
                                 ENCODED_SPEED_DIM + \
                                 ENCODED_ROUTE_DIM, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, n_actions)
            )
        
        # self.linear1 = nn.Linear(hidden_features + \
        #                          ENCODED_SPEED_DIM + \
        #                          ENCODED_ROUTE_DIM, 16)
        # self.relu3 = nn.ReLU()
        # self.linear2 = nn.Linear(16, n_actions)
        
    def forward(self, batch):
        if isinstance(batch, tuple):
            graph_batch, speed_batch, route_batch = batch
        else:
            raise(Exception(f"Data must be a tuple, got {type(batch)}"))
            
        x = self.conv1(graph_batch.x, graph_batch.edge_index, graph_batch.edge_weight)
        x = self.batch_norm1(x)
        x = self.relu1(x)
        x = self.conv2(x, graph_batch.edge_index, graph_batch.edge_weight)
        x = self.batch_norm2(x)
        x = self.relu2(x)
        x = self.global_pooling(x, graph_batch.batch)
        
        v = self.speed_encoder(speed_batch)
        r = self.route_encoder(route_batch.permute(0, 2, 1))
        
        h = torch.cat((x,v,r), dim=1)
        h = self.output_layer(h)
        return h.squeeze()
    
    @property
    def num_actions(self):
        return self._n_actions
    
    @property
    def num_node_features(self):
        return self._n_features
    

def init_agent(n_actions, node_features, hidden_features, obs_type):
    if obs_type == 'gcn':
        agent = GCNPolicy(node_features, 
                          n_actions,
                          hidden_features=hidden_features)
    elif obs_type == 'gcn_speed':
        agent = GCNPolicySpeed(node_features, 
                          n_actions,
                          hidden_features=hidden_features)
    elif obs_type == 'gcn_speed_route':
        agent = GCNPolicySpeedRoute(node_features, 
                          n_actions,
                          hidden_features=hidden_features)
    else:
        raise Exception(f"Unexpected argument 'obs_type'. Value: {obs_type}, expected 'gcn', 'gcn_speed' or 'gcn_speed_route'")
    return agent      
        
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
    







    
