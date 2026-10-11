"""Print the layout of PAMFN-style feature files so --in_dim / --clip-num can be checked before training.

    $MLAVL_PY scripts/inspect_features.py /data/mlavl/features/RG
    $MLAVL_PY scripts/inspect_features.py /data/mlavl/features/FISV
"""
import glob
import os
import sys

import numpy as np


def main(root):
    for f in sorted(glob.glob(os.path.join(root, '*.npy'))):
        d = np.load(f, allow_pickle=True)
        if d.dtype == object:
            d = d.item()
            keys = list(d.keys())
            lens = [d[k].shape[0] for k in keys]
            print(f'{os.path.basename(f)}: dict with {len(keys)} videos, feature shape {d[keys[0]].shape}, '
                  f'T min/median/max = {min(lens)}/{sorted(lens)[len(lens) // 2]}/{max(lens)}, e.g. key {keys[0]!r}')
        else:
            print(f'{os.path.basename(f)}: array {d.shape} {d.dtype}')
    for f in sorted(glob.glob(os.path.join(root, '*.txt'))):
        with open(f) as fh:
            lines = fh.read().splitlines()
        print(f'{os.path.basename(f)}: {len(lines) - 1} rows, header: {lines[0]!r}, first: {lines[1]!r}')


if __name__ == '__main__':
    main(sys.argv[1])
