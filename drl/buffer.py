# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 15:41:15 2023

@author: lucac
"""

from collections import deque
import random
from torch_geometric.data import Data
from torch_geometric.data import Batch
import torch


class ReplayBuffer:
    def __init__(self, size, force_cpu=False):
        self._memory = deque(maxlen=size)
        self.device = "cuda" if torch.cuda.is_available() and not force_cpu else "cpu"
        
    def __len__(self):
        return len(self._memory)

    def add_experience(self, state, action, reward, new_state, done):
        self._memory.append((state, action, reward, new_state, done))  
        
    def sample(self, batch_size):
        samples = random.sample(self._memory, batch_size)
        
        # Create batched tensors 
        action = torch.tensor([el[1] for el in samples],
                              device=self.device)
        reward = torch.tensor([el[2] for el in samples], 
                              dtype=torch.float32,
                              device=self.device)
        done = torch.tensor([el[4] for el in samples], 
                            dtype=torch.bool,
                            device=self.device)
        if isinstance(samples[0][0], torch.Tensor) \
            or isinstance(samples[0][0], np.ndarray):
            state = torch.tensor([el[0] for el in samples], 
                                 dtype=torch.float32,
                                 device=self.device)
            new_state = torch.tensor([el[3] for el in samples], 
                                     dtype=torch.float32,
                                     device=self.device)
        elif isinstance(samples[0][0], torch_geometric.data.Data):
            state = Batch.from_data_list([el[0] for el in samples])
            new_state = Batch.from_data_list([el[3] for el in samples])
        else:
            raise Exception(f"Unexpected type {type(minibatch[0][0])}")
            
        return [state, action, reward, new_state, done]
        
        
if __name__ == "__main__":
    import numpy as np
    import random
    import torch_geometric
    from torch_geometric.utils.random import erdos_renyi_graph
    # import sys
    # sys.path.insert(0, '..')
    # from policy import GCNPolicy
        
    from torch_geometric.nn import GCNConv
    from torch_geometric.nn import aggr
    
    
    def set_seed(seed):
        torch_geometric.seed_everything(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        
    def random_graph(node_features, num_nodes, edge_prob=0.5):
        edge_list = erdos_renyi_graph(num_nodes, edge_prob)
        nodes = torch.rand(num_nodes, node_features)
        return Data(x=nodes, edge_index=edge_list)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    class GCNPolicy(torch.nn.Module):
        def __init__(self, n_features, n_actions):
            super().__init__()
            self.n_features = n_features
            self.n_actions = n_actions
            
            self.conv1 = GCNConv(n_features, 2)
            self.relu1 = torch.nn.ReLU()
            self.conv2 = GCNConv(2, 3)
            self.relu2 = torch.nn.ReLU()
            
            self.global_pooling = aggr.MaxAggregation()
            self.linear1 = torch.nn.Linear(3, n_actions)
            
        def forward(self, batch):
            x = self.conv1(batch.x, batch.edge_index, batch.edge_weight)
            x = self.relu1(x)
            x = self.conv2(x, batch.edge_index, batch.edge_weight)
            x = self.relu2(x)
            x = self.global_pooling(x, batch.batch)
            x = self.linear1(x)
            return x.squeeze()
        
    class MLPPolicy(torch.nn.Module):
        def __init__(self, n_features, n_actions):
            super().__init__()
            self.linear1 = torch.nn.Linear(n_features, n_actions)
            
        def forward(self, x):
            return self.linear1(x)
        
    N_FEATURES = 4
    N_ACTIONS = 2
    BATCH_SIZE = 3
    BUFFER_SIZE = 8 
    gamma = 1.0
    
    set_seed(4)
    buffer = ReplayBuffer(BUFFER_SIZE)
    
    # graphs = [random_graph(N_FEATURES, np.random.randint(2, 4)) for _ in range(BUFFER_SIZE)]
    # new_graphs = [random_graph(N_FEATURES, np.random.randint(2, 4)) for _ in range(BUFFER_SIZE)]
    # for i in range(BUFFER_SIZE):
    #     action = np.random.randint(0, 3)
    #     reward = np.random.rand()
    #     done = (np.random.rand() > 0.5)
    #     buffer.add_experience(graphs[i], action, reward, new_graphs[i], done)
    
    # minibatch = buffer.sample(BATCH_SIZE)
    
    # state, action, reward, new_state, done = minibatch
    # model = GCNPolicy(N_FEATURES, N_ACTIONS)
    
    # Q = model(new_state)
    # q_hat, _ = torch.max(Q, axis=1)
    # y = reward + done * gamma * q_hat
    # L = torch.mean((model(state)[range(BATCH_SIZE), action] - y)**2)
    
    # TEST WITH ARRAY ENVIRONMENT
    buffer = ReplayBuffer(BUFFER_SIZE)
    state = [np.random.rand(N_FEATURES) for _ in range(BUFFER_SIZE)]
    new_state = [np.random.rand(N_FEATURES) for _ in range(BUFFER_SIZE)]
    for i in range(BUFFER_SIZE):
        action = np.random.randint(N_ACTIONS)
        reward = np.random.rand()
        done = (np.random.rand() > 0.5)
        buffer.add_experience(state[i], action, reward, new_state[i], done)
    
    minibatch = buffer.sample(BATCH_SIZE)
    state, action, reward, new_state, done = minibatch
    model = MLPPolicy(N_FEATURES, N_ACTIONS)
    
    GAMMA = 0.98
    
    def q_loss1(model, minibatch, batch_size):
        # Compute loss                
        L = 0.0
        s_, a_, r_, s_new_, done_ = minibatch
        for i in range(batch_size):
            s, a, r, s_new, done = s_[i], a_[i], r_[i], s_new_[i], done_[i]
            # q_hat = self.policy(s_new)[action]
            q_hat = torch.max(model(s_new))
            
            if done:
                y = r
            else:
                # action = torch.argmax(self.policy(s_new))
                y = r + GAMMA * q_hat
            print("r", r, "q_hat", q_hat, "done", done)
            print("y", y)
            # print(model(s)[a])
            L += (model(s)[a] - y)**2
        L /= batch_size
        return L
    
    def q_loss2(model, minibatch, batch_size):
        state, action, reward, new_state, done = minibatch
        
        q_hat, _ = torch.max(model(new_state), axis=1)
        y = reward + ~done * GAMMA * q_hat
        
        L = torch.mean((model(state)[range(batch_size), action] - y)**2)
        return L
    
    L1 = q_loss1(model, minibatch, BATCH_SIZE)
    print("\n"*1)
    L2 = q_loss2(model, minibatch, BATCH_SIZE)
    print(L1, L2)
    
    ## TODO testare su cartpole e vedere se è più efficiente

    
    
    
    
    