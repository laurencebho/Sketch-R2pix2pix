'''
for generating a dataset - moves all relevant images (images from the sketchyGAN dataset) into 'train' directory
'''

import os
import shutil


valid_categories = [
    'airplane',  'ant',  'apple',  'banana',  'bear', 'bee',  'bell',  'bench',  'bicycle',  'candle',  'cannon', 'car_(sedan)',
    'castle',  'cat', 'chair',  'church',  'couch',  'cow',  'cup',  'dog',  'elephant',  'geyser',  'giraffe',
    'hammer',  'hedgehog',  'horse',  'hotdog',  'hourglass',  'jellyfish',  'knife',  'lion',  'motorcycle',
    'mushroom',  'pig',  'pineapple', 'pizza', 'pretzel', 'rifle', 'scissors', 'scorpion', 'sheep', 'snail',
    'spoon', 'starfish', 'strawberry', 'tank', 'teapot', 'tiger', 'volcano', 'zebra'
]

base_dir = 'datasets/tx_000000000000'

os.mkdir(f'{base_dir}/train')

subdirs = next(os.walk(base_dir))[1]
for subdir in subdirs:
    subdir_name = subdir.split('/')[-1]
    if subdir_name in valid_categories:
        for filename in os.listdir(f'{base_dir}/{subdir}'):
            if filename.endswith('.jpg'):
                shutil.move(f'{base_dir}/{subdir}/{filename}', f'{base_dir}/train/{filename}')