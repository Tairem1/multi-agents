@ECHO 
  
python train.py --seed 0 --wandb --log --learning_rate 1e-04 --hidden_features 32
python train.py --seed 0 --wandb --log --learning_rate 1e-04 --hidden_features 64

PAUSE