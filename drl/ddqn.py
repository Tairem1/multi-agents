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
from torch_geometric.data import Data

# from timer import Timer
# t = Timer()

class DDQN:
    def __init__(self, 
                 env,
                 policy,
                 test_env=None,
                 device='cpu',
                 replay_buffer_size=20_000,
                 gamma = 0.95,
                 learning_rate = 1e-03,
                 eps_start = 0.9,
                 eps_min = 0.01,
                 eps_decay = 0.99,
                 start_learn=100,
                 batch_size=16,
                 episodes_per_epoch = 5,
                 test_every = 5_000,
                 network_update_frequency=100):
        self._memory = deque(maxlen=replay_buffer_size)
        self.gamma = gamma
        self.epsilon_start = eps_start
        self.epsilon_min = eps_min
        self.epsilon_decay = eps_decay
        self.learning_rate = learning_rate
        self.episodes_per_epoch = episodes_per_epoch
        self.batch_size = batch_size
        self.start_learn = max(start_learn, batch_size)
        self.network_update_frequency = network_update_frequency
        self.test_every = test_every
        
        # Create network and target network
        self.policy = policy
        self.target_policy = copy.deepcopy(policy)
        self.env = env
        self.test_env = test_env
        
        # Deep learning parameters
        print(self.policy)
        print(self.policy.parameters())
        self.optimizer = torch.optim.Adam(self.policy.parameters(),
                                          lr=self.learning_rate)
        self._device = device
        
        
    def learn(self, 
              total_timesteps,
              log = False,
              epoch_callback = None,
              checkpoint_callback = None,
              episode_callback = None,
              loss_callback = None,
              eval_callback = None):
        # Perform initial training setup
        self._setup_train()
        
        # Start new epoch and first episode
        epoch_count = 0
        episode_reward = 0.0
        self._start_new_epoch()
        state, _ = self._start_new_episode(self.env)
        
        # Training main loop
        for count in range(total_timesteps):
            if checkpoint_callback is not None:
                checkpoint_callback(self.policy)
            
            # Update e-greedy value
            self._update_epsilon(count)   
            
            # Gather and store new transition in replay buffer
            action = self._eps_greedy_action_selection(state)
            new_state, reward, done, truncated, info = self._step(action)
            self._add_experience(state, action, reward, new_state, done)
            state = new_state
            episode_reward += reward
            
            if done:    # End of episode
                self._cumulative_epoch_reward += episode_reward
                self._episode_print(episode_reward, info) if log else None
                
                if episode_callback is not None:
                    episode_callback()
                
                if self._end_of_epoch():
                    # End of epoch
                    epoch_count += 1
                    avg_epoch_reward = self._cumulative_epoch_reward/self.episodes_per_epoch
                    self._epoch_print(avg_epoch_reward, count, total_timesteps) 
                    
                    if epoch_callback is not None:
                        epoch_callback(avg_epoch_reward, 
                                       epoch_count, 
                                       self.policy) 
                        
                    self._start_new_epoch()
                    
                state, _ = self._start_new_episode(self.env)
                episode_reward = 0.0
                
            # Update weights
            self._update_network_weights(count, loss_callback)
            
            if self.test_env is not None:
                if (count % self.test_every) == 0:
                    mean, std = self._evaluate_model()
                    print(f"\tEVAL: Episode reward: {mean:.2f} +- {std:.2f}")
                    eval_callback(mean, std, count, self.policy)
            
        self.env.close()
        
    def _evaluate_model(self, seed=0, n_episodes=5):
        rewards = []
        with torch.no_grad():
            self.policy.eval()
            for i in range(n_episodes):
                total_reward = 0.0
                state, _ = self._start_new_episode(self.test_env, seed=i)
                for _ in range(10000):
                    action = int(torch.argmax(self.policy(state)))
                    state, reward, done, _, _ = self._step(action)
                    total_reward += reward
                    if done:
                        break
                rewards.append(total_reward)
        self.policy.train()
        mean, std = np.mean(rewards), np.std(rewards)
        return mean, std
        
                    
                    
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
                
    def _update_epsilon(self, count):
        self.epsilon = self.epsilon_min + (self.epsilon_start - self.epsilon_min) * \
            np.exp(-1. * count / self.epsilon_decay)
        
    def _epoch_print(self, avg_epoch_reward, count, total_timesteps):                    
        print(f"\tTRAIN {float(count)*100/total_timesteps}%: {self.episodes_per_epoch} episodes average reward: {avg_epoch_reward}")
    
    def _episode_print(self, episode_reward, info):                    
        print(f"\tTRAIN: Episode reward: {episode_reward}, {info}, {self.epsilon}")

    def _end_of_epoch(self):
        return (self._episode_count % self.episodes_per_epoch) == 0
                
    def _setup_train(self):
        self.policy.train()         
        self.target_policy.eval()
    
    def _start_new_episode(self, env, seed=None):
        self._episode_count += 1
        self._episode_reward = 0.0
        
        if seed is None:
            state, info = env.reset()
        else:
            state, info = env.reset(seed=seed)
        
        if isinstance(state, np.ndarray):
            state = torch.tensor(state).to(self._device)
        elif isinstance(state, Data):
            state = state.to(self._device)
        else:
            raise Exception("Unexpected state type")
        
        return state, info
    
    def _start_new_epoch(self):
        self._episode_count = 0
        self._cumulative_epoch_reward = 0.0
    
    def _q_loss(self, minibatch):
        # Compute loss                
        L = 0.0
        for s, a, r, s_new, done in minibatch:
            if done:
                y = torch.tensor(r, device=self._device)
            else:
                # XXX ACTIVATE LINE action = torch.argmax(self.target_policy(s_new))
                action = torch.argmax(self.policy(s_new))
                q_hat = self.policy(s_new)[action]
                y = torch.tensor(r, device=self._device) + self.gamma * q_hat
                
            L += (self.policy(s)[a] - y)**2
        L /= len(minibatch)
        return L
    
    def _step(self, action):
        new_state, reward, done, truncated, info = self.env.step(action)
        if isinstance(new_state, np.ndarray):
            new_state = torch.tensor(new_state, device=self._device)
        return new_state.to(self._device), reward, done, truncated, info
        
    def _update_network_weights(self, count, loss_callback):
        if len(self._memory) > self.start_learn: 
            # LEARN PHASE
            self.optimizer.zero_grad()
            minibatch = random.sample(self._memory, self.batch_size)
            loss = self._q_loss(minibatch)
            loss.backward()
            
            if loss_callback is not None:
                loss_callback(loss, count)
            self.optimizer.step()
            
            # XXX REMOVE COMMENTS
            # Update target network 
            # if count % self.network_update_frequency == 0:
            #     self._update_target_policy()
        
        
        
        
                    
            
                
            
                
                
            
            
            
            
            