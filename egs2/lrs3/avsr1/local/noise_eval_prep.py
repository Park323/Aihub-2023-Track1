#!/usr/bin/env python

# Copyright 2023  Jeongkyun Park
#           2023  Sogang University
# Apache 2.0

import tqdm
import os
import os.path as osp
import random
import logging
import argparse

import soundfile
import numpy as np


class Utils:
    @staticmethod
    def get_parser():
        """Returns the Parser object required to take inputs to data_prep.py"""
        parser = argparse.ArgumentParser(
            description="LRS-3 Noise Preparation for Evaluation",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument("--seed", type=int, default=42, help="Random seed")
        parser.add_argument("--test_path", type=str, help="Path to the Test files")
        parser.add_argument("--noise_scp", type=str, help="Path to the Noise scp files")
        
        return parser


def seed_all(seed:int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    

def fit_random_noise(noise_path, nsamples):
    with soundfile.SoundFile(noise_path) as f:
        if f.frames == nsamples:
            noise = f.read(dtype=np.float64)
        elif f.frames < nsamples:
            offset = np.random.randint(0, nsamples - f.frames)
            # noise: (Time, Nmic)
            noise = f.read(dtype=np.float64)
            # Repeat noise
            noise = np.pad(
                noise,
                [(offset, nsamples - f.frames - offset), (0, 0)],
                mode="wrap",
            )
        else:
            offset = np.random.randint(0, f.frames - nsamples)
            f.seek(offset)
            # noise: (Time, Nmic)
            noise = f.read(nsamples, dtype=np.float64)
            if len(noise) != nsamples:
                raise RuntimeError(f"Something wrong: {noise_path}")
    return noise


def generate_noises(test_dir, utt_ids, wav_paths, noise_paths):
    noise_save_dir = osp.join(test_dir, "noise")
    os.makedirs(noise_save_dir, exist_ok=True)
    
    pbar = tqdm.tqdm(zip(utt_ids, wav_paths))
    for utt_id, wav_path in pbar:
        pbar.set_description(f"[{utt_id}]")
        
        sf = soundfile.SoundFile(wav_path)
        noise_path = np.random.choice(noise_paths)
        noise = fit_random_noise(noise_path, nsamples=len(sf))
        noise_sr = sf.samplerate
        sf.close()

        noise_save_path = osp.join(noise_save_dir, utt_id+".wav")
        soundfile.write(noise_save_path, data=noise, samplerate=noise_sr)


def main():
    parser = Utils.get_parser()
    args = parser.parse_args()
    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=logfmt)

    seed_all(args.seed)

    with open(osp.join(args.test_path, "wav.scp")) as f:
        utt_ids = []
        wav_paths = []
        for line in f:
            utt_id, wav_path = line.strip().split()
            utt_ids.append(utt_id)
            wav_paths.append(wav_path)

    with open(args.noise_scp) as f:
        noise_paths = [path.strip() for path in f.readlines()]

    generate_noises(args.test_path, utt_ids, wav_paths, noise_paths)


if __name__ == "__main__":
    main()
