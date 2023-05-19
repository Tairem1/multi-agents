# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 17:10:09 2023

@author: lucac
"""

import os
import argparse
import torch
import pickle
import wandb
import json

from scenes import Scene
from policy import init_agent
from drl.ddqn import DDQN
from util.utils import set_seed
from util.callbacks import EpochCallback,CheckpointCallback,EvalCallback,LossCallback
    
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', type=str, default="dqn_gcn",
                        help="Model name.")
    parser.add_argument('--group_name', type=str, default=None,
                        help="Specify an experiment group name to group together multiple runs")
    parser.add_argument('--log', action="store_true", default=False) 
    
    parser.add_argument('--seed', type=int, default=0,
                        help="Seed for random initialisation.")             
    parser.add_argument('--wandb', action="store_true", default=False)  
    parser.add_argument('--policy_network', type=str, default="gcn",
                        help="Type of policy network. Can be of type 'gcn', 'gcn_speed', 'gcn_speed_route")
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=32, 
                        help="Minibatch Size.")
    parser.add_argument('--gamma', type=float, default=0.99,
                        help="discount factor.")
    parser.add_argument('--total_timesteps', type=int, default=100_000,
                        help="The total of number of samples to train on.")
    parser.add_argument('--learning_rate', type=float, default=1e-04,
                        help="learning rate for optimizer.")
    parser.add_argument('--hidden_features', type=int, default=16,
                        help="Number of features in the hidden layers.")
    parser.add_argument('--network_update_frequency', type=int, default=100,
                        help="update target network every.")
    args = parser.parse_args()
    return args
    
def create_checkpoint_directory():
    if not os.path.isdir("./checkpoint/"):
        os.mkdir("./checkpoint/")
    for i in range(1000):
        checkpoint_dir = f"./checkpoint/{args.tag}_{i:03d}/"
        if not os.path.isdir(checkpoint_dir):
            os.mkdir(checkpoint_dir)
            return checkpoint_dir
    raise Exception("Unable to create checkpoint directory.")
    
def save_args(checkpoint_dir, args): 
    with open(checkpoint_dir+'args.pkl', 'wb') as fp:
        pickle.dump(args, fp)


if __name__ == "__main__":
    args = parse()
    checkpoint_dir = create_checkpoint_directory()
    config = {
        'EPISODES_PER_EPOCH': 40,  
        'BATCH_SIZE': args.batch_size,
        'GAMMA': args.gamma,
        'EPS_START': 1.0,
        'EPS_END': 0.00,
        'EPS_DECAY': int(2*args.total_timesteps/4),
        'TOTAL_TIMESTEPS': args.total_timesteps,
        'TEST_EVERY': args.total_timesteps//20,
        'REPLAY_BUFFER_SIZE': 100_000,
        'LR': args.learning_rate,
        'NETWORK_UPDATE_FREQUENCY': args.network_update_frequency,
        'START_LEARN': 1_000,
        'RANDOM_SEED': args.seed,
        'CHECKPOINT_DIR': checkpoint_dir,
        'HIDDEN_FEATURES': args.hidden_features,
        'N_TESTING_EPISODES': 40,
        'POLICY_NETWORK': args.policy_network # Can be 'gcn', 'gcn_speed', 'gcn_speed_route'
        }
    
    print(args)
    print(config)
    print(f'Saving configuration data to: {checkpoint_dir}')
    save_args(checkpoint_dir, config)
    
    if args.wandb:
        wandb.init(project="GCN-DRL", group="Intersection", job_type="train")
        wandb.config.update(config)
    
    set_seed(args.seed)
    
    # ENVIRONMENT SETUP
    dt = 0.2 # time steps in terms of seconds. In other words, 1/dt is the FPS.
    # The world is 120 meters by 120 meters. ppm is the pixels per meter.
    env = Scene(dt, 
                  width = 120, 
                  height = 120, 
                  ppm = 5, 
                  render=True,
                  testing=False,
                  discrete_actions=True,
                  window_name="Training Environment",
                  seed=args.seed,
                  obs_type=args.policy_network)
    env.load_scene("scene01")
    
    # test_env = Scene(dt, 
    #               width = 120, 
    #               height = 120, 
    #               ppm = 5, 
    #               render=True,
    #               testing=True,
    #               discrete_actions=True,
    #               window_name="Testing Environment",
    #               seed=1234,
    #               obs_type=args.policy_network)
    # test_env.load_scene("scene01")
    test_env = None
    
    
    # PYTORCH MODEL AND OPTIMIZER SETUP
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = init_agent(Scene.ACTION_SIZE, Scene.OBS_SIZE, 
                       args.hidden_features, args.policy_network).to(device)
    optimizer = torch.optim.Adam(agent.parameters(),
                               config['LR'])
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, 
                                      total_iters=config['TOTAL_TIMESTEPS'],
                                      power=2.0)
    scheduler = None
    model = DDQN(env, 
                  agent, 
                  optimizer,
                  scheduler,
                  test_env=test_env, 
                  device=device,
                  batch_size=config['BATCH_SIZE'], 
                  start_learn = config['START_LEARN'],
                  eps_start = config['EPS_START'],
                  eps_min=config['EPS_END'],
                  eps_decay=config['EPS_DECAY'],
                  test_every=config['TEST_EVERY'],
                  replay_buffer_size=config['REPLAY_BUFFER_SIZE'], 
                  network_update_frequency=config['NETWORK_UPDATE_FREQUENCY'],
                  episodes_per_epoch=config['EPISODES_PER_EPOCH'])
     
    
    # START TRAINING    
    print('*'*30)
    print("Training initiating....")
    print()
    
    try:
        model.learn(total_timesteps=config['TOTAL_TIMESTEPS'], 
                    log=args.log, 
                    epoch_callback=EpochCallback(checkpoint_dir,wandb_log=args.wandb),
                    checkpoint_callback=CheckpointCallback(checkpoint_dir, save_every=args.total_timesteps//20),
                    loss_callback=LossCallback(wandb_log=args.wandb),
                    eval_callback=EvalCallback(checkpoint_dir, wandb_log=args.wandb),
                    n_testing_episodes=config['N_TESTING_EPISODES'])
    finally:
        env.close()
        test_env.close()
        wandb.finish()
    