import os
import pickle

base_dir = 'datasets/sketchy'
test_files = os.listdir(f'{base_dir}/test')
train_files = os.listdir(f'{base_dir}/train')


with open(f'datasets/sketchy.pkl', 'rb') as fh:
    saved = pickle.load(fh)
    svgs = saved['sketches']
    svg_names = list(svgs.keys())

unmatched = []
for tf in train_files:
    valid = False
    for s in svg_names:
        if s.startswith(tf):
            valid = True
            break
    if not valid:
        unmatched.append(tf)
    

for name in unmatched:
    print(name)