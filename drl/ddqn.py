# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 16:40:47 2023

@author: lucac
"""

import random
import numpy as np
from collections import deque

import copy
import torch

class DDQN:
    def __init__(self, 
                 env,
                 policy,
                 replay_buffer_size=20_000,
                 gamma = 0.95,
                 learning_rate = 1e-03,
                 eps_min = 0.01,
                 eps_decay = 0.99,
                 episodes_per_epoch = 100,
                 start_learn=100,
                 batch_size=16,
                 network_update_frequency=1_000):
        self._memory = deque(maxlen=replay_buffer_size)
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = eps_min
        self.epsilon_decay = eps_decay
        self.learning_rate = learning_rate
        self.episodes_per_epoch = episodes_per_epoch
        self.start_learn = start_learn
        self.batch_size = batch_size
        self.network_update_frequency = network_update_frequency
        
        # Create network and target network
        self.policy = policy
        self.target_policy = copy.deepcopy(policy)
        self.env = env
        
        # Deep learning parameters
        self.optimizer = torch.optim.Adam(self.policy.parameters(),
                                          lr=self.learning_rate)
        
    def learn(self, 
              N_iter=int(100_000),
              log = True):
        # Perform initial training setup
        self._setup_train()
        
        # Start new epoch and first episode
        self._start_new_epoch()
        state = self._start_new_episode()
        
        # Start training iterations
        for count in range(N_iter):
            self._update_epsilon()      # Update e-greedy value
            
            # Gather and store new transition
            action = self._eps_greedy_action_selection(state)
            new_state, reward, done, info = self.env.step(action)
            self._add_experience(state, action, reward, new_state, done)
            
            # Reset environment if episode is finished
            if done:
                if log:
                    print(info)
                    print(f"Episode reward: {self.env.episode_reward}")
                state = self._start_new_episode()
                self.epoch_reward += self.env.episode_reward
                
                if self._end_of_epoch():
                    if log:
                        self._print_reward(self.epoch_reward)
                    self._start_new_epoch()
                
            if len(self._memory) > self.start_learn: 
                # LEARN PHASE
                self.optimizer.zero_grad()
                minibatch = random.sample(self._memory, self.batch_size)
                
                loss = self._q_loss(minibatch)
                loss.backward()
                self.optimizer.step()
                
                # Update target network
                if count % self.network_update_frequency == 0:
                    self._update_target_policy()
                    
                    
                    
    ######################
    ### HELPER METHODS ###
    ######################
    def _add_experience(self, old_state, action, reward, new_state, done):
        self._memory.append((old_state, action, reward, new_state, done))
        
    def _update_target_policy(self):
        self.target_policy.set_weights(self.policy.get_weights())
        
    def _eps_greedy_action_selection(self, current_state):
        with torch.no_grad():
            p = np.random.rand()
            if p < self.epsilon:
                # Random action
                action = np.random.choice(self.policy.num_actions)
            else:
                # Action selection based on Q function
                action = int(torch.argmax(self.policy(current_state)))
        return action
                
    def _update_epsilon(self):
        # Update epsilon in epsilon-greedy exploration
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.epsilon_min)
        
    def _print_reward(self, epoch_reward):                    
        print(f"\tTRAIN: Cumulative reward: {epoch_reward/self.episodes_per_epoch}")
        
    def _end_of_epoch(self):
        return (self._episode_count % self.episodes_per_epoch) == 0
                
    def _setup_train(self):
        self.policy.train()         
        self.target_policy.eval()
    
    def _start_new_episode(self):
        self._episode_count += 1
        self._episode_reward = 0.0
        return self.env.reset()
    
    def _start_new_epoch(self):
        self._episode_count = 0
        self.epoch_reward = 0.0
    
    def _q_loss(self, minibatch):
        # Compute loss                
        L = 0.0
        for s, a, r, s_new, done in minibatch:
            if done:
                y = r
            else:
                y = r + self.gamma * torch.argmax(self.target_policy(s_new))
            L += (self.policy(s)[a] - y)**2
        L /= len(minibatch)
        return L
        
        
        
                    
            
                
            
                
                
            
            
            
            
            