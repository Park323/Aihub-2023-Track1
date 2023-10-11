import torch
import queue
import os
import random
import warnings
import time
import json
import argparse
from glob import glob

import subprocess

from espnet2.tasks.asr import ASRTask

import nova
from nova import DATASET_PATH


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

    config = args.parse_args()
    warnings.filterwarnings('ignore')

    if config.mode == 'train':
        subprocess.run("cd egs2/aihub2023/asr1")
        subprocess.run(f"PYTHONPATH='../../..' bash asr.sh --stage 1")
