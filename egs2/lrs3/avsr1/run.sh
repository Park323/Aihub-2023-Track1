#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


train_set="train"
valid_set="dev"
test_sets="test dev"


avsr_tag=train_avsr_conformer
avsr_config=conf/train_avsr_conformer.yaml
lm_config=conf/train_lm.yaml  # Not Used, as use_lm=false

./avsr.sh \
    --skip_data_prep false \
    --skip_train false \
    --skip_eval false \
    --stage 1 \
    --lang en \
    --ngpu 1 \
    --nj 16 \
    --inference_nj 16 \
    --nbpe 5000 \
    --max_wav_duration 30 \
    --feats_normalize global_mvn \
    --audio_format "wav" \
    --audio_feats_type raw \
    --visual_format "mp4" \
    --visual_feats_type raw \
    --use_lm false \
    --avsr_tag "${avsr_tag}" \
    --lm_config ${lm_config} \
    --avsr_config "${avsr_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text" "$@"
