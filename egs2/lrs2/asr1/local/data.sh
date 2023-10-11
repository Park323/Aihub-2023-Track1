#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

. ./db.sh
. ./path.sh
. ./cmd.sh

# Make the Folders where ESPNet data-prep files will be stored
for dataset in train dev test; do
    log "Creating the ./data/${dataset} folders"
    mkdir -p ./data/${dataset}
done

# generate the utt2spk, wav.scp and text files
log "Generating the utt2spk, wav.scp and text files"
python3 ./local/data_prep.py --pretrain_path ${LRS2}/pretrain --train_path ${LRS2}/train --val_path ${LRS2}/train --test_path ${LRS2}/test

log "Generating the spk2utt files"
utils/utt2spk_to_spk2utt.pl data/train/utt2spk > data/train/spk2utt
utils/utt2spk_to_spk2utt.pl data/dev/utt2spk > data/dev/spk2utt
utils/utt2spk_to_spk2utt.pl data/test/utt2spk > data/test/spk2utt

log "Fix sorting issues by calling fix_data_dir.sh"
utils/fix_data_dir.sh data/train
utils/fix_data_dir.sh data/test
utils/fix_data_dir.sh data/dev

log "Validate the data directory"
utils/validate_data_dir.sh data/train --no-feats
utils/validate_data_dir.sh data/test --no-feats
utils/validate_data_dir.sh data/dev --no-feats
