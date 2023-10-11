#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


train_set="train"
valid_set="dev"
test_sets="test"


vsr_tag=
vsr_config=
use_lm=false
lm_config=conf/train_lm.yaml  # Not Used, as use_lm=false


feats_normalize=global_mvn


inference_args=


./vsr.sh \
    --skip_data_prep false \
    --skip_train false \
    --skip_eval false \
    --stage 1 \
    --lang en \
    --ngpu 1 \
    --nj 32 \
    --inference_nj 16 \
    --nbpe 5000 \
    --max_mp4_duration 24 \
    --visual_format "mp4" \
    --feats_type raw \
    --feats_normalize "${feats_normalize}" \
    --use_lm "${use_lm}" \
    --vsr_tag "${vsr_tag}" \
    --lm_config ${lm_config} \
    --vsr_config "${vsr_config}" \
    --inference_args "${inference_args}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text" "$@"
