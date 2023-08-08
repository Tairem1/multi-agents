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
    parser.add_argument('--render', action="store_true", default=False)
    parser.add_argument('--seed', type=int, default=0,
                        help="Seed for random initialisation.")             
    parser.add_argument('--wandb', action="store_true", default=False,
                        help="Log run on wandb.")  
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=32, 
                        help="Minibatch Size.")
    parser.add_argument('--gamma', type=float, default=0.99,
                        help="discount factor.")
    parser.add_argument('--total_timesteps', type=int, default=100_000,
                        help="The total of number of samples to train on.")
    parser.add_argument('--replay_buffer_size', type=int, default=10_000,
                        help="The size of the replay buffer in DQN.")
    parser.add_argument('--learning_rate', type=float, default=1e-04,
                        help="learning rate for optimizer.")
    parser.add_argument('--network_update_frequency', type=int, default=1_000,
                        help="update target network every.")
    
    # Network architecture
    parser.add_argument('--hidden_features', type=int, default=16,
                        help="Number of features in the hidden layers.")
    parser.add_argument('--gcn_conv_layer', type=str, default='GCN',
                        help="GCN convolution operation: {GCN, GraphSAGE}")
    parser.add_argument('--n_conv_layers', type=int, default=2,
                        help="Number of convolution operations.")
    parser.add_argument('--policy_network', type=str, default="gcn",
                        help="Type of policy network. Can be of type 'gcn', 'gcn_speed', 'gcn_speed_route")
    
    
    # Reward parameters
    parser.add_argument('--reward_parameters', '--list', nargs='+',
                        default=[-1.0, +1.0, -1.0, 0.03, 0.02, 0.01, 0.02],
                        type=float,
                        help="Reward parameters to be passed as a list (timeout, goal_reached, collision, velocity, action, idle, proximity).")
    
    args = parser.parse_args()
    return args
    
def create_checkpoint_directory():
    if not os.path.isdir("./checkpoint/"):
        os.mkdir("./checkpoint/")
    if not os.path.isdir(f"./checkpoint/{args.tag}/"):
        os.mkdir(f"./checkpoint/{args.tag}")
    for i in range(1000):
        checkpoint_dir = f"./checkpoint/{args.tag}/{i:03d}/"
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
    
    reward_configuration = {
            'timeout': args.reward_parameters[0],
            'goal_reached': args.reward_parameters[1],
            'collision': args.reward_parameters[2],
            'velocity': args.reward_parameters[3],
            'action': args.reward_parameters[4],
            'idle': args.reward_parameters[5],
            'proximity': args.reward_parameters[6],
    }
    print(reward_configuration)
    
    dt = 0.2 # time steps in terms of seconds. In other words, 1/dt is the FPS.
    config = {
        'EPISODES_PER_EPOCH': int(args.total_timesteps*40/100_000),  
        'BATCH_SIZE': args.batch_size,
        'GAMMA': args.gamma,
        'EPS_START': 0.5,
        'EPS_END': 0.05,
        'EPS_DECAY': int(2*args.total_timesteps/4),
        'TOTAL_TIMESTEPS': args.total_timesteps,
        'TEST_EVERY': args.total_timesteps//20,
        'REPLAY_BUFFER_SIZE': args.replay_buffer_size,
        'LR': args.learning_rate,
        'NETWORK_UPDATE_FREQUENCY': args.network_update_frequency,
        'START_LEARN': args.total_timesteps//100,
        'RANDOM_SEED': args.seed,
        'CHECKPOINT_DIR': checkpoint_dir,
        'HIDDEN_FEATURES': args.hidden_features,
        'N_TESTING_EPISODES': 100,
        'POLICY_NETWORK': args.policy_network, # Can be 'gcn', 'gcn_speed', 'gcn_speed_route'
        'REWARD': reward_configuration,
        'GCN_CONV_LAYER': args.gcn_conv_layer,
        'N_CONV_LAYERS': args.n_conv_layers,
        'dt': dt
        }
    
    print(args)
    print(config)
    print(f'Saving configuration data to: {checkpoint_dir}')
    save_args(checkpoint_dir, config)
    
    if args.wandb:
        run = wandb.init(project="GCN-DRL", group=args.group_name, job_type="train")
        wandb.config.update(config)
        print(run.name)
    
    set_seed(args.seed)
    
    # ENVIRONMENT SETUP
    # The world is 120 meters by 120 meters. ppm is the pixels per meter.
    env = Scene(dt, 
                  width = 120, 
                  height = 120, 
                  ppm = 5, 
                  render=args.render,
                  testing=False,
                  discrete_actions=True,
                  window_name="Training Environment",
                  seed=args.seed,
                  obs_type=args.policy_network,
                  reward_configuration=reward_configuration,
                  agent=args.agent)
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
    agent = init_agent(Scene.ACTION_SIZE, 
                       Scene.OBS_SIZE, 
                       args.hidden_features, 
                       args.policy_network,
                       n_conv_layers=args.n_conv_layers, 
                       gcn_conv_layer=args.gcn_conv_layer
                       ).to(device)
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
    