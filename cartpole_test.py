# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 15:13:52 2023

@author: AdminS
"""

import gymnasium as gym
import torch
import numpy as np

from cartpole import MLPPolicy

if __name__ == "__main__":
    try:
        env = gym.make('CartPole-v1', render_mode="human")
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # model_path = "./checkpoint/cartpole_000/model_10000.pth"
        model_path = "./checkpoint/cartpole_005/reward_best.pth"
        agent = MLPPolicy(state_size, action_size)
        agent.load_state_dict(torch.load(model_path))
        agent = agent.to(device)
        
        N_EVALUATION_EPISODES = 5
        rewards = []
        for i in range(N_EVALUATION_EPISODES):
            state, info = env.reset(seed=i)
            state = torch.tensor(state).to(device)
            
            done = False
            episode_reward = 0.0
            
            while not done:
                action = agent.act(state)
                
                state, reward, terminated, truncated, info = env.step(action)
                state = torch.tensor(state).to(device)
                
                episode_reward += reward
                
                done = (terminated or truncated)
            rewards.append(episode_reward)
        
        print(f"MEAN REWARD: {np.mean(rewards)}")
        print(f"REWARD STD: {np.std(rewards)}")
    finally:
        env.close()
    
    
            
            
        
        
    
    
    