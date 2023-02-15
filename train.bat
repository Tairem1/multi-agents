@ECHO 
set group="Test DQN-GCN"
  
python train.py --tag gcn_test --seed 0 --group_name %group% ^
  --batch_size 16 --gamma 0.99 --learning_rate 0.0001 ^
  --total_timesteps 100_000 --log
  
PAUSE