#!/bin/sh
#SBATCH --job-name=rter # Job name
#SBATCH --ntasks=4 # Run on a single CPU
#SBATCH --time=23:50:00 # Time limit hrs:min:sec
#SBATCH --output=test_job%j.out # Standard output and error log
#SBATCH --gres=gpu:1
#SBATCH --partition=cl1_48h-1G


python3 main.py --hidden_dim 300 --max_window_size 40 --epochs 50 --lr 1e-2 --dataset MELD --dropout .5 --num_layers 1
