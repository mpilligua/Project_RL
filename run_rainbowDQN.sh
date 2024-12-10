#!/bin/bash
#SBATCH -n 2 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -D /fhome/pmlai10/Project_RL # working directory
#SBATCH -p dcc
#SBATCH --mem 2048 # 2GB solicitados.
#SBATCH -o /ghome/mpilligua/RL/Project_RL/error_folder/%x_%u_%j.out # File to which STDOUT will be written
#SBATCH -e /ghome/mpilligua/RL/Project_RL/error_folder/%x_%u_%j.err # File to which STDERR will be written
#SBATCH --gres gpu:1 # Para pedir gráficas

python3 /ghome/mpilligua/RL/Project_RL/freeway/rainbow_dqn.py