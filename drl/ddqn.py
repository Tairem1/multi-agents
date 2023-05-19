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

from drl.buffer import ReplayBuffer

import time
class Timer:
    def __init__(self):
        pass
    
    def start(self):
        self.start_time = time.time()
        
    def stop(self, function_name):
        self.stop_time = time.time()
        print(f"TIMER\t{function_name} {self.stop_time - self.start_time}")
t = Timer()

class DDQN:
    def __init__(self, 
                 env,
                 policy,
                 optimizer,
                 lr_scheduler=None,
                 test_env=None,
                 device='cpu',
                 replay_buffer_size=20_000,
                 gamma=0.95,
                 eps_start=0.9,
                 eps_min=0.01,
                 eps_decay=0.99,
                 start_learn=100,
                 batch_size=16,
                 episodes_per_epoch=10,
                 test_every=5_000,
                 network_update_frequency=100):
        self._memory = ReplayBuffer(replay_buffer_size, device)
        self.gamma = gamma
        self.epsilon_start = eps_start
        self.epsilon_min = eps_min
        self.epsilon_decay = eps_decay
        self.episodes_per_epoch = episodes_per_epoch
        self.batch_size = batch_size
        self.start_learn = max(start_learn, batch_size)
        self.network_update_frequency = network_update_frequency
        self.test_every = test_every
        
        # Create network and target network
        self.policy = policy
        self.target_policy = copy.deepcopy(policy).to(device)
        self.env = env
        self.test_env = test_env
        
        # Deep learning parameters
        self.optimizer = optimizer
        self.scheduler = lr_scheduler
        self._device = device
        
        
    def learn(self, 
              total_timesteps,
              log = False,
              epoch_callback = None,
              checkpoint_callback = None,
              episode_callback = None,
              loss_callback = None,
              eval_callback = None,
              n_testing_episodes = 10):
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
            new_state, reward, terminated, truncated, info = self.env.step(action)
            
            done = terminated or truncated
            self._memory.add_experience(state, action, reward, new_state, done)
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
                    mean, std = self._evaluate_model(n_testing_episodes)
                    print(f"\tEVAL: Episode reward: {mean:.2f} +- {std:.2f}")
                    eval_callback(mean, std, count, self.policy)
            
        self.env.close()
                    
    ######################
    ### HELPER METHODS ###
    ######################
    def _eps_greedy_action_selection(self, current_state):
        with torch.no_grad():
            p = np.random.rand()
            if p < self.epsilon:
                # Random action
                action = np.random.randint(self.policy.num_actions)
            else:
                # # Action selection based on Q function
                # if isinstance(current_state, np.ndarray):
                #     s = torch.tensor(current_state, device=self._device)
                # elif isinstance(current_state, Data):
                #     s = copy.deepcopy(current_state).to(self._device)
                # elif isinstance(current_state, tuple):
                #     if len(current_state) == 2:
                #         graph_batch, v_batch = current_state
                #         g = copy.deepcopy(graph_batch).to(self._device)
                #         v = torch.tensor([[v_batch]], device=self._device)
                #         s = (g, v)
                #     elif len(current_state) == 3:
                #         graph_batch, v_batch = current_state
                #         g = copy.deepcopy(graph_batch).to(self._device)
                #         v = torch.tensor([[v_batch]], device=self._device)
                # else:
                #     raise Exception(f"Unsupported state type: {type(current_state)}")
                s = self._move_state_to_device(current_state)
                self.policy.eval()
                action = int(torch.argmax(self.policy(s)))
                self.policy.train()
        return action
                
    def _update_epsilon(self, count):
        eps =  self.epsilon_start + \
             (self.epsilon_min - self.epsilon_start)*count/self.epsilon_decay
        self.epsilon = max(self.epsilon_min, eps)
                             
    def _epoch_print(self, avg_epoch_reward, count, total_timesteps):  
        if self.scheduler is not None:                  
            print(f"\tTRAIN {float(count)*100/total_timesteps}%: {self.episodes_per_epoch} episodes average reward: {avg_epoch_reward}, eps: {self.epsilon:.2f}, lr: {self.scheduler.get_last_lr()}, {len(self.scheduler.get_last_lr())}")
        else:                 
            print(f"\tTRAIN {float(count)*100/total_timesteps}%: {self.episodes_per_epoch} episodes average reward: {avg_epoch_reward}, eps: {self.epsilon:.2f}")
            
    
    def _episode_print(self, episode_reward, info):  
        if self.scheduler is not None:                  
            print(f"\tTRAIN: Episode reward: {episode_reward}, {info}, eps: {self.epsilon}, lr: {self.scheduler.get_last_lr()}")
        else:
            print(f"\tTRAIN: Episode reward: {episode_reward}, {info}, eps: {self.epsilon}")

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
        
        return state, info
    
    def _start_new_epoch(self):
        self._episode_count = 0
        self._cumulative_epoch_reward = 0.0
    
    def _q_loss(self, minibatch):
        state, action, reward, new_state, done = minibatch
        
        q_hat, _ = torch.max(self.target_policy(new_state), axis=1)
        y = reward + ~done * self.gamma * q_hat
        
        L = torch.mean((self.policy(state)[range(self.batch_size), action] - y)**2)
        return L
        
    def _update_network_weights(self, count, loss_callback):
        if len(self._memory) > self.start_learn: 
            # LEARN PHASE
            self.optimizer.zero_grad()
            minibatch = self._memory.sample(self.batch_size)
            loss = self._q_loss(minibatch)
            loss.backward()
            
            if loss_callback is not None:
                loss_callback(loss, count)
            self.optimizer.step()
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Update target network 
            if count % self.network_update_frequency == 0:
                self.target_policy.load_state_dict(self.policy.state_dict())
        
    def _evaluate_model(self, n_testing_episodes=5):
        rewards = []
        with torch.no_grad():
            self.policy.eval()
            self.test_env.reset_rng()
            for i in range(n_testing_episodes):
                total_reward = 0.0
                state, info = self.test_env.reset(seed=i)
                state = self._move_state_to_device(state)
                
                for _ in range(10000):
                    action = int(torch.argmax(self.policy(state)))
                                
                    state, reward, terminated, truncated, info = self.test_env.step(action)
                    state = self._move_state_to_device(state)
                    
                    total_reward += reward
                    if terminated or truncated:
                        break
                rewards.append(total_reward)
        self.policy.train()
        mean, std = np.mean(rewards), np.std(rewards)
        return mean, std
    
    def _move_state_to_device(self, state):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state).to(self._device)
        elif isinstance(state, Data):
            state = state.to(self._device)
        elif isinstance(state, tuple):
            if len(state) == 2:
                graph_batch, v_batch = state
                g = copy.deepcopy(graph_batch).to(self._device)
                v = torch.tensor([[v_batch]], device=self._device)
                state = (g, v)
            elif len(state) == 3:
                graph_batch, v_batch, route_batch = state
                g = copy.deepcopy(graph_batch).to(self._device)
                v = torch.tensor([[v_batch]], device=self._device, dtype=torch.float32)
                r = torch.tensor(np.array([route_batch]), device=self._device, dtype=torch.float32)
                state = (g, v, r)
        else:
            raise Exception("Unexpected state type")
        return state
        

        
        
        
        
                    
            
                
            
                
                
            
            
            
            
            