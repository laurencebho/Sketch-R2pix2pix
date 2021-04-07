#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --gres=gpu
#SBATCH -p ug-gpu-small
#SBATCH --qos=short
#SBATCH -t 07-00:00:00
#SBATCH --job-name=sketchr2pix2pixtests
#SBATCH --mem=24G
#SBATCH --mail-user laurence.ho@durham.ac.uk
#SBATCH --mail-type=ALL

source ~/pix2pixenv/bin/activate
module load cuda/10.1-cudnn7.6

python3 sketch-r2pix2pix_train.py  --category_loss_off --dataroot ./datasets/sketchy --name sr2pix2pix_no_classifier_loss--model sketchr2pix2pix --direction AtoB --display_id=0 --input_nc 8 --output_nc 3 --save_epoch_freq 2

python3 sketch-r2pix2pix_train.py --discrim_layers 8 --dataroot ./datasets/sketchy --name sr2pix2pix_core_8_layers--model sketchr2pix2pix --direction AtoB --display_id=0 --input_nc 8 --output_nc 3 --save_epoch_freq 2

python3 sketch-r2pix2pix_train.py --discrim_layers 6 --dataroot ./datasets/sketchy --name sr2pix2pix_core_6_layers--model sketchr2pix2pix --direction AtoB --display_id=0 --input_nc 8 --output_nc 3 --save_epoch_freq 2

python3 sketch-r2pix2pix_train.py --discrim_layers 3 --dataroot ./datasets/sketchy --name sr2pix2pix_core_3_layers--model sketchr2pix2pix --direction AtoB --display_id=0 --input_nc 8 --output_nc 3 --save_epoch_freq 2