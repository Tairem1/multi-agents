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
                 start_learn=100,
                 batch_size=16,
                 episodes_per_epoch = 100,
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
              total_timesteps,
              log = True,
              epoch_callback = None,
              episode_callback = None):
        # Perform initial training setup
        self._setup_train()
        
        # Start new epoch and first episode
        epoch_count = 0
        self._start_new_epoch()
        state = self._start_new_episode()
        
        # Training main loop
        for count in range(total_timesteps):
            
            # Update e-greedy value
            self._update_epsilon()     
            
            # Gather and store new transition in replay buffer
            action = self._eps_greedy_action_selection(state)
            new_state, reward, done, info = self.env.step(action)
            self._add_experience(state, action, reward, new_state, done)
            
            if done:
                self._cumulative_epoch_reward += self.env.episode_reward
                self._episode_print(info) if log else None
                
                # End of episode
                if episode_callback is not None:
                    episode_callback()
                
                if self._end_of_epoch():
                    # End of epoch
                    epoch_count += 1
                    avg_epoch_reward = self._cumulative_epoch_reward/self.episodes_per_epoch
                    self._epoch_print(avg_epoch_reward) if log else None
                    
                    if epoch_callback is not None:
                        epoch_callback(avg_epoch_reward, 
                                       epoch_count, 
                                       self.policy) 
                        
                    self._start_new_epoch()
                    
                state = self._start_new_episode()
                
            # Update weights
            self._update_network_weights(count)
                    
                    
                    
    ######################
    ### HELPER METHODS ###
    ######################
    def _add_experience(self, old_state, action, reward, new_state, done):
        self._memory.append((old_state, action, reward, new_state, done))
        
    def _update_target_policy(self):
        self.target_policy.load_state_dict(self.policy.state_dict())
        
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
        
    def _epoch_print(self, avg_epoch_reward):                    
        print(f"\tTRAIN: {self.episodes_per_epoch} episodes average reward: {avg_epoch_reward}")
    
    def _episode_print(self, info):                    
        print(f"Episode reward: {self.env.episode_reward}, {info['end_reason']}")

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
        self._cumulative_epoch_reward = 0.0
    
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
    
    def _update_network_weights(self, count):
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
        
        
        
        
                    
            
                
            
                
                
            
            
            
            
            