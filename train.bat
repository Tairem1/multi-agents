@ECHO 
set group="Reward tuning"


::(timeout, goal_reached, collision, velocity, action, idle, proximity)
::(-1.0,    +1.0,         -1.0,      0.03,     0.02,   0.01, 0.02)
python train.py --seed 0 --log --learning_rate 1e-05 --hidden_features 64^
     --total_timesteps 100_000 --tag 05_25 --policy_network gcn_speed_route ^
     --batch_size 512 --network_update_frequency 1_000 --wandb --group_name=%group%^
     --reward_parameters -1.0 1.0 -1.0 0.03 0.02 0.01 0.02
     
python train.py --seed 0 --log --learning_rate 1e-05 --hidden_features 64^
     --total_timesteps 100_000 --tag 05_25 --policy_network gcn_speed_route ^
     --batch_size 512 --network_update_frequency 100 --wandb --group_name=%group%^
     --reward_parameters -1.0 1.0 -1.0 0.03 0.02 0.01 0.02
     
python train.py --seed 0 --log --learning_rate 1e-05 --hidden_features 64^
     --total_timesteps 100_000 --tag 05_25 --policy_network gcn_speed_route ^
     --batch_size 512 --network_update_frequency 10 --wandb --group_name=%group%^
     --reward_parameters -1.0 1.0 -1.0 0.03 0.02 0.01 0.02
     
     
PAUSE