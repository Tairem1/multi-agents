@ECHO 

python train.py --seed 0 --log --learning_rate 1e-05 --hidden_features 64 --total_timesteps 1_000_000 --tag 02_03 --policy_network gcn_speed_route --wandb --batch_size 512 --network_update_frequency 20000  
python train.py --seed 0 --log --learning_rate 1e-05 --hidden_features 64 --total_timesteps 1_000_000 --tag 02_03 --policy_network gcn_speed_route --wandb --batch_size 512 --network_update_frequency 40000  



PAUSE