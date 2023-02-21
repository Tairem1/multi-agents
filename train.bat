@ECHO 
::set group="Test DQN-GCN"
  
python cartpole.py --seed 0 --wandb
python cartpole.py --seed 1 --wandb
python cartpole.py --seed 2 --wandb

PAUSE