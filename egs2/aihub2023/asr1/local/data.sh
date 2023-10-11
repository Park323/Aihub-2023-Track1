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

. ./path.sh
. ./cmd.sh


if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 1
fi

# Make the Folders where ESPNet data-prep files will be stored
for dataset in train dev; do
    log "Creating the ./data/${dataset} folders"
    mkdir -p ./data/${dataset}
done

# generate the utt2spk, wav.scp and text files
log "Generating the utt2spk, wav.scp and text files"
python3 ./local/data_prep.py

log "Generating the spk2utt files"
utils/utt2spk_to_spk2utt.pl data/train/utt2spk > data/train/spk2utt
utils/utt2spk_to_spk2utt.pl data/dev/utt2spk > data/dev/spk2utt

log "Fix sorting issues by calling fix_data_dir.sh"
utils/fix_data_dir.sh data/train
utils/fix_data_dir.sh data/dev

log "Validate the data directory"
utils/validate_data_dir.sh data/train --no-feats
utils/validate_data_dir.sh data/dev --no-feats