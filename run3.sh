#!/bin/sh
#SBATCH --job-name=rter # Job name
#SBATCH --ntasks=4 # Run on a single CPU
#SBATCH --time=23:50:00 # Time limit hrs:min:sec
#SBATCH --output=test_job%j.out # Standard output and error log
#SBATCH --gres=gpu:1
#SBATCH --partition=cl1_48h-1G

printf "\n\nLearning Rate\n\n"

python3 main.py --hidden_dim 200 --max_window_size 20 --epochs 100 --lr 5e-5 --dataset MELD --dropout .3 --print_f1
printf "\n\n\n\n"

python3 main.py --hidden_dim 200 --max_window_size 20 --epochs 100 --lr 1e-4 --dataset MELD --dropout .3 --print_f1
printf "\n\n\n\n"

python3 main.py --hidden_dim 200 --max_window_size 20 --epochs 100 --lr 2e-4 --dataset MELD --dropout .3 --print_f1