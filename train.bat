@ECHO 

python train.py --seed 0 --log --learning_rate 1e-05 --hidden_features 64^
     --total_timesteps 100_000 --tag 05_19 --policy_network gcn_speed_route ^
     --batch_size 512 --network_update_frequency 1_000  

PAUSE