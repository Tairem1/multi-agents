@ECHO 
set group="Mixed Traffic"

::(timeout, goal_reached, collision, velocity, action, idle, proximity)
::(-1.0,    +1.0,         -1.0,      0.03,     0.02,   0.01, 0.02)
     
python multiagent_train.py --seed 0 --log --learning_rate 1e-05 --hidden_features 64^
     --tag svo --policy_network gcn_speed_route --batch_size 512 ^
     --group_name=%group% --n_conv_layers 2 --gcn_conv_layer GraphSAGE^
     --reward_parameters 0  0   -2.0    0.05    0    0    0 ^
     --total_timesteps 500_000 --replay_buffer_size 200_000 --render ^
     --agent vehicle --pedestrian_level L0 --vehicle_level L1 ^
     --pedestrian_agent_id 000 --svo 90
          
::python train.py --seed 0 --log --learning_rate 1e-05 --hidden_features 64^
  ::   --tag reward_tuning --policy_network gcn_speed_route --batch_size 512 ^
    :: --group_name=%group% --n_conv_layers 2 --gcn_conv_layer GraphSAGE^
     ::--reward_parameters 0  0   -2.5    0.05    0    0    0  --wandb^
     ::--total_timesteps 2_000_000 --replay_buffer_size 400_000  
     
PAUSE