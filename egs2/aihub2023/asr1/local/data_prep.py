#!/usr/bin/env python

# Copyright 2023  Jeongkyun Park
#           2023  Sogang University
# Apache 2.0


import re
import argparse
import logging
import os
import glob
import pdb;pdb.set_trace()
import tqdm
from pathlib import Path
import subprocess
from typing import List, Optional, Union

import numpy as np

try:
    from nova import DATASET_PATH
except:
    DATASET_PATH='data/sample'

from espnet2.text.korean_separator import char2grp


class Utils:
    @staticmethod
    def unzip_groups(transcript):
        # Use the latter one, which is grammatically correct
        pattern = '(\(([^(/)]+)\)?([^(/)]*))\/?(\(?([^(/)]+)\)(\3)?)'
        if re.search(pattern, transcript):
            _transcript = re.sub(pattern, f"\{2}", transcript)
            _transcript = re.sub('[(/)]', "", _transcript)
            result = Utils.unzip_groups(_transcript)
        else:
            result = transcript
        return result
    
    @staticmethod
    def refine_text(text: str) -> str:
        text_val = text.strip()
        # text_val = re.sub('\xa0',' ',text_val) # \xa0 : space
        # text_val = re.sub('[Xx]',' ',text_val) # x : mute
        # text_val = Utils.unzip_groups(text_val) # Cases : (A)/(B) (A)(B) (A)a/(B)a
        # text_val = re.sub('[/\u2028\u2029]', '', text_val) # Unusual line/paragraph separators
        return text_val

    @staticmethod
    def save_list_to_file(list_data: list, save_path: str) -> None:
        """ "Writes content of list_data to a file, line-by-line

        Args:
        list_data: List of Text to be saved to the text file
        save_path: file to save the list_data
        """
        with open(save_path, "w") as f:
            for line in list_data:
                f.write(line + "\n")

    @staticmethod
    def get_parser():
        """Returns the Parser object required to take inputs to data_prep.py"""
        parser = argparse.ArgumentParser(
            description="AIHUB2023 Data Preparation steps",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        
        return parser


class DatasetUtils:
    @staticmethod
    def train_val_files(
        train_path: str, train_val_ratio: float = 0.96, random_seed: int = 0,
    ) -> Union[List[str], List[str], List[str], List[str]]:
        """Splits the files in 'train_path' into the train set and test set,
           and returns the full Train/Validation files.

        Args:
        train_val_path (str): Path to the Folder with the Train/Val data
        train_val_ratio (float): Ratio of the Train/Test file ratio
        random_seed (int): Seed for the file shufling

        Returns:
        train_files (list) : Paths of Training Data
        val_files (list) : Paths of Validation Data
        """
        saved_train_paths = os.path.join(train_path, "train_subset")
        saved_val_paths = os.path.join(train_path, "val_subset")
        
        if os.path.exists(saved_train_paths) and os.path.exists(saved_val_paths):
            logging.info("Use the last train/val split logs")
            with open(saved_train_paths, "r") as f:
                train_labels = [line.strip() for line in f.readlines()]
            with open(saved_val_paths, "r") as f:
                val_labels = [line.strip() for line in f.readlines()]
        else:
            logging.info("Save train/val split logs")
            
            label_path = os.path.join(train_path, "train_label")
        
            with open(label_path) as f:
                lines = f.readlines()[1:]
            
            labels = [x.strip() for x in lines]
            
            np.random.seed(random_seed)
            np.random.shuffle(labels)
            
            num_train = int(train_val_ratio * len(labels))
            train_labels = labels[0:num_train]
            val_labels = labels[num_train:]
            
            with open(saved_train_paths, "w") as f:
                [f.write(line+"\n") for line in train_labels]
            with open(saved_val_paths, "w") as f:
                [f.write(line+"\n") for line in val_labels]

        train_root = os.path.join(train_path, "train_data")
        
        train_filenames = []
        train_transcriptions = []
        for label in train_labels:
            filename, transcription= label.split(',')
            train_filenames.append(filename)
            train_transcriptions.append(transcription)
        
        val_filenames = []
        val_transcriptions = []
        for label in train_labels:
            filename, transcription= label.split(',')
            val_filenames.append(filename)
            val_transcriptions.append(transcription)
        
        return train_filenames, train_transcriptions, val_filenames, val_transcriptions

    @staticmethod
    def generate_espnet_data(
        filepaths: list, labels: list, dataset: str
    ) -> Union[List[str], List[str], List[str], List[str]]:
        """Generates the utt2spk, text, mp4 and wav data required by ESPNET

        Args:
        speaker_folders (list): The folders from where to extract data
        dataset (str): The dataset we are working with (train, test, dev)

        Returns:
        utt2spk (list) : Utterence to Speaker data
        text (list) : Utterence to Transcript data
        wav (list) : Utterence to Wav-Path data
        """
        utt2spk = []
        text = []
        text_sep = []
        wav = []

        for uid, (filepath, label) in tqdm.tqdm(enumerate(zip(filepaths, labels))):
            spk_id = "0"
            utt_id = f"{dataset}_{uid}"
            utt2spk.append(utt_id + " " + spk_id)
            wav.append(utt_id + " " + os.path.join(DATASET_PATH, "train_data", filepath))
            label = Utils.refine_text(label)
            text.append(utt_id + " " + label)
            text_sep.append(utt_id + " " + char2grp(label))

        return utt2spk, text, text_sep, wav

    @staticmethod
    def perform_data_prep(filepaths: list, labels: list, dataset: str) -> None:
        """Performs ESPNET related Data-Preparation.
        Generates the utt2spk, text, text_sep, mp4.scp and wav.scp files

        Args:
        speaker_folders (list): The folders from where to extract data
        dataset (str): The dataset we are working with (train, test, dev)
        """
        utt2spk, text, text_sep, wav = DatasetUtils.generate_espnet_data(filepaths, labels, dataset)

        utt2spk_file = os.path.join("data", dataset, "utt2spk")
        text_file = os.path.join("data", dataset, "text")
        text_sep_file = os.path.join("data", dataset, "text_sep")
        wav_file = os.path.join("data", dataset, "wav.scp")

        Utils.save_list_to_file(utt2spk, utt2spk_file)
        Utils.save_list_to_file(text, text_file)
        Utils.save_list_to_file(text_sep, text_sep_file)
        Utils.save_list_to_file(wav, wav_file)


def main():
    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=logfmt)

    train_files, train_labels, dev_files, dev_labels = DatasetUtils.train_val_files(DATASET_PATH)
    
    logging.info(f"Performing Data Preparation for TRAIN")
    DatasetUtils.perform_data_prep(train_files, train_labels, "train")

    logging.info(f"Performing Data Preparation for DEV")
    DatasetUtils.perform_data_prep(dev_files, dev_labels, "dev")
    
    import shutil
    shutil.copytree("data/dev", "data/test")


if __name__ == "__main__":
    main()
