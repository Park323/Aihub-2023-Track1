import subprocess

import torch

from espnet2.tasks.asr import ASRTask


def run_perl(input_path, output_path):
    # Run the Perl script using subprocess.run
    with open(input_path, 'r') as input_file, open(output_path, 'w') as output_file:
        return subprocess.run(['utils/utt2spk_to_spk2utt.pl'], stdin=input_file, stdout=output_file, check=True)


def train_prep(config):

    # Whole process ver.
    from sys import path
    cmd = f"bash run.sh --nbpe {config.nbpe} --asr_config {config.config} --asr_tag aihub --stop_stage 10"
    if config.kor_sep:
        cmd += " --bpe_train_text data/train/text_sep"
        cmd += " --lm_train_text data/train/text_sep" # Fix this to use LM
    if config.args:
        cmd += f" {config.args}"
    subprocess.run(cmd.split(), 
                   env=dict(PYTHONPATH=":".join([*path, "../../.."])))
