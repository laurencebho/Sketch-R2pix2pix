#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --gres=gpu
#SBATCH -p ug-gpu-small
#SBATCH --qos=short
#SBATCH -t 01-00:00:00
#SBATCH --job-name=pix2pixLaurie
#SBATCH --mem=22G
#SBATCH --mail-user laurence.ho@durham.ac.uk
#SBATCH --mail-type=ALL

#source ~/pix2pixenv/bin/activate
#module load cuda/10.1-cudnn7.6

#python3 train.py --dataroot ./datasets/sketchy --name sketchy_pix2pix --model sketchr2pix2pix --direction AtoB --display_id=0 --input_nc 8 --output_nc 3 --gpu_ids -1
python3 train.py --dataroot ./datasets/sketchy --name sketchy_pix2pix --model sketchr2pix2pix --direction AtoB --display_id=0 --input_nc 8 --output_nc 3

#python test.py --dataroot ./datasets/sketchy --name sketchy_pix2pix --model pix2pix --direction AtoB --input_nc 8 --output_nc 3
