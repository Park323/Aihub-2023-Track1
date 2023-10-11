import os
# import sys
# root = os.getcwd()
os.chdir("egs2/aihub2023/asr1")
# local = os.getcwd()
# sys.path.insert(0, local)
# sys.path.insert(0, root)

import warnings
import argparse
from glob import glob


try:
    import nova
    from nova import DATASET_PATH
except:
    DATASET_PATH = "data/sample"

with open("db.sh", "w") as f:
    f.write(f"AIHUB2023={DATASET_PATH}\n")

from train import train


def inference(path, model, **kwargs):
    model.eval()

    results = []
    for i in glob(os.path.join(path, '*')):
        results.append(
            {
                'filename': i.split('/')[-1],
                'text': single_infer(model, i)[0]
            }
        )
    return sorted(results, key=lambda x: x['filename'])


if __name__ == '__main__':

    args = argparse.ArgumentParser()

    # DONOTCHANGE
    args.add_argument('--mode', type=str, default='train')
    args.add_argument('--iteration', type=str, default='0')
    args.add_argument('--pause', type=int, default=0)
    
    args.add_argument('--config', type=str, required=True)
    args.add_argument('--nbpe', type=int, required=True)
    args.add_argument('--stage', type=int)
    args.add_argument('--stop_stage', type=int)

    config = args.parse_args()
    warnings.filterwarnings('ignore')

    if config.mode == 'train':
        train(config)