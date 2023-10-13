import os
import sys
os.chdir("egs2/aihub2023/asr1")
import logging
import warnings
import argparse
from glob import glob

import torch

from train import train_prep
from inference import inference
from espnet2.tasks.asr import ASRTask

try:
    import nova
    from nova import DATASET_PATH
except:
    nova = None
    DATASET_PATH = "data/sample"

print("DATASET_PATH :", DATASET_PATH)

with open("db.sh", "w") as f:
    f.write(f"AIHUB2023={DATASET_PATH}\n")


def bind_model(model, optimizer=None):
    def save(path, *args, **kwargs):
        state = {'model': model.state_dict()}
        if optimizer:
            state.update({'optimizer': optimizer.state_dict()})
        torch.save(state, os.path.join(path, 'model.pt'))
        print(f'NOVA saved the model in `{path}`')

    def load(path, *args, **kwargs):
        state = torch.load(os.path.join(path, 'model.pt'))
        model.load_state_dict(state['model'])
        if 'optimizer' in state and optimizer:
            optimizer.load_state_dict(state['optimizer'])
        print(f'Model loaded from {path}')

    # 추론
    def infer(path, **kwargs):
        return inference(path, model)

    if nova:
        nova.bind(save=save, load=load, infer=infer)  # 'nova.bind' function must be called at the end.
        logging.info("NOVA successfully binded the model")


if __name__ == '__main__':

    args = argparse.ArgumentParser()

    # DONOTCHANGE
    args.add_argument('--mode', type=str, default='train')
    args.add_argument('--iteration', type=str, default='0')
    args.add_argument('--pause', type=int, default=0)
    
    args.add_argument('--config', type=str, required=True)
    args.add_argument('--nbpe', type=int, required=True)
    args.add_argument('--token_type', type=str, required=True)
    args.add_argument('--kor_sep', action='store_true')
    args.add_argument('--args', type=str)

    config = args.parse_args()
    warnings.filterwarnings('ignore')

    if config.mode == 'train':
        train_prep(config)

    # Build model
    if config.token_type == "bpe":
        stats_dir = f"asr_stats_raw_kr_bpe{config.nbpe}"
        spm_dir   = f"data/kr_token_list/{"sep" if config.kor_sep else "raw"}/bpe_unigram{config.nbpe}"
    else:
        stats_dir = f"asr_stats_raw_kr_{config.token_type}"
        spm_dir   = f"data/kr_token_list/{config.token_type}"
    text_file = "text_sep" if config.kor_sep else "text"
    cmd = f"""
        --config {config.config} --output_dir exp/asr_aihub --frontend_conf fs=16k --use_preprocessor true 
        --token_type {config.token_type} --bpemodel {spm_dir}/bpe.model --token_list {spm_dir}/tokens.txt 
        --non_linguistic_symbols none --cleaner none --g2p none 
        --train_data_path_and_name_and_type dump/raw/train/wav.scp,speech,sound --train_shape_file exp/{stats_dir}/train/speech_shape 
        --train_data_path_and_name_and_type dump/raw/train/{text_file},text,text --train_shape_file exp/{stats_dir}/train/text_shape.bpe
        --valid_data_path_and_name_and_type dump/raw/dev/wav.scp,speech,sound --valid_shape_file exp/{stats_dir}/valid/speech_shape 
        --valid_data_path_and_name_and_type dump/raw/dev/{text_file},text,text --valid_shape_file exp/{stats_dir}/valid/text_shape.bpe 
        --ignore_init_mismatch false --ngpu {torch.cuda.device_count()} --multiprocessing_distributed True --resume false"""
    parser = ASRTask.get_parser()
    args = parser.parse_args(args=cmd.split())
    model = ASRTask.build_model(args=args)
    
    # AIHUB2023. Bind model
    bind_model(model=model)

    if config.mode == "train":
        ASRTask.main(args, model=model)
        
        print("Train Finished")
    
        if nova:
            nova.paused(scope=locals())
    