import subprocess

import torch

from espnet2.tasks.asr import ASRTask


def run_perl(input_path, output_path):
    # Run the Perl script using subprocess.run
    with open(input_path, 'r') as input_file, open(output_path, 'w') as output_file:
        return subprocess.run(['utils/utt2spk_to_spk2utt.pl'], stdin=input_file, stdout=output_file, check=True)


def train(config):

    # Whole process ver.
    from sys import path
    cmd = f"bash run.sh --nbpe {config.nbpe} --asr_config {config.config} --asr_tag aihub --stop_stage 10"
    if config.kor_sep:
        cmd += f" --bpe_train_text data/train/text_sep"
    if config.args:
        cmd += f" {config.args}"
    subprocess.run(cmd.split(), 
                   env=dict(PYTHONPATH=":".join([*path, "../../.."])))
    
    # Step 11
    stats_dir = f"asr_stats_raw_kr_bpe{config.nbpe}"
    text_file = "text_sep" if config.kor_sep else "text"
    cmd = f"""
        --use_preprocessor true --bpemodel data/kr_token_list/bpe_unigram{config.nbpe}/bpe.model 
        --token_type bpe --token_list data/kr_token_list/bpe_unigram{config.nbpe}/tokens.txt --non_linguistic_symbols none --cleaner none --g2p none 
        --config {config.config} --output_dir exp/asr_aihub --frontend_conf fs=16k 
        --train_data_path_and_name_and_type dump/raw/train/wav.scp,speech,sound --train_shape_file exp/{stats_dir}/train/speech_shape 
        --train_data_path_and_name_and_type dump/raw/train/{text_file},text,text --train_shape_file exp/{stats_dir}/train/text_shape.bpe
        --valid_data_path_and_name_and_type dump/raw/dev/wav.scp,speech,sound --valid_shape_file exp/{stats_dir}/valid/speech_shape 
        --valid_data_path_and_name_and_type dump/raw/dev/{text_file},text,text --valid_shape_file exp/{stats_dir}/valid/text_shape.bpe 
        --ignore_init_mismatch false --ngpu {torch.cuda.device_count()} --multiprocessing_distributed True --resume false"""
    parser = ASRTask.get_parser()
    args = parser.parse_args(args=cmd.split())
    ASRTask.main(args)
    
    print("Train Finished")