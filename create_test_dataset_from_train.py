'''
run this to move some files (2 per category) from
datasets/sketchy/train to datasets/sketchy/test
'''
import os
import shutil

base_dir = 'datasets/sketchy'

if not os.path.exists(f'{base_dir}/test'):
    os.makedirs(f'{base_dir}/test')

filenames = sorted(os.listdir(f'{base_dir}/train'))

for i in range(0, len(filenames), 100):
    filenames_to_move = [filenames[i], filenames[i+1], filenames[i+2]]

    for filename in filenames_to_move:
        shutil.move(f'{base_dir}/train/{filename}', f'{base_dir}/test/{filename}')

print('test DS created')