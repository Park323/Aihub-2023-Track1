import subprocess

import torch

# from local.data_prep import main as prepare_data
# from espnet2.tasks.asr import ASRTask

def run_perl(input_path, output_path):
    # Run the Perl script using subprocess.run
    with open(input_path, 'r') as input_file, open(output_path, 'w') as output_file:
        return subprocess.run(['utils/utt2spk_to_spk2utt.pl'], stdin=input_file, stdout=output_file, check=True)


def train(config):
    
    # Whole process ver.
    from sys import path
    cmd = f"bash run.sh --asr_config {config.config}"
    if config.nbpe:
        cmd += f"--nbpe {config.nbpe}"
    if config.stage:
        cmd += f"--stage {config.stage}"
    subprocess.run(cmd.split(), 
                   env=dict(PYTHONPATH=":".join([*path, "../../.."])))
    
    # # Separated processes ver.
    
    # # Step 1
    # prepare_data()
    
    # print("Generating the spk2utt files")
    # run_perl("data/train/utt2spk", "data/train/spk2utt")
    # run_perl("data/dev/utt2spk", "data/dev/spk2utt")
    
    # print("Fix sorting issues by calling fix_data_dir.sh")
    # subprocess.run(["utils/fix_data_dir.sh", "data/train"])
    # subprocess.run(["utils/fix_data_dir.sh", "data/dev"])

    # print("Validate the data directory")
    # subprocess.run(["utils/validate_data_dir.sh", "data/train", "--no-feats"])
    # subprocess.run(["utils/validate_data_dir.sh", "data/dev", "--no-feats"])
    
    # # Step 2
    
    # # Step 3
    
    # # Step 4
    
