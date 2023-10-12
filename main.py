import os
os.chdir("egs2/aihub2023/asr1")

import warnings
import argparse
from glob import glob

try:
    import nova
    from nova import DATASET_PATH
except:
    DATASET_PATH = "data/sample"

print("DATASET_PATH :", DATASET_PATH)

with open("db.sh", "w") as f:
    f.write(f"AIHUB2023={DATASET_PATH}\n")

from train import train


if __name__ == '__main__':

    args = argparse.ArgumentParser()

    # DONOTCHANGE
    args.add_argument('--mode', type=str, default='train')
    args.add_argument('--iteration', type=str, default='0')
    args.add_argument('--pause', type=int, default=0)
    
    args.add_argument('--config', type=str, required=True)
    args.add_argument('--nbpe', type=int, required=True)
    args.add_argument('--kor_sep', action='store_true')
    args.add_argument('--args', type=str)

    config = args.parse_args()
    warnings.filterwarnings('ignore')

    if config.mode == 'train':
        train(config)
