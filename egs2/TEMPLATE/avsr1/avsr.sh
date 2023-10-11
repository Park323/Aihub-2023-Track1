#!/usr/bin/env bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
min() {
  local a b
  a=$1
  for b in "$@"; do
      if [ "${b}" -le "${a}" ]; then
          a="${b}"
      fi
  done
  echo "${a}"
}

SECONDS=0

# General configuration
stage=1              # Processes starts from the specified stage.
stop_stage=10000     # Processes is stopped at the specified stage.
skip_stages=         # Spicify the stage to be skipped
skip_data_prep=false # Skip data preparation stages.
skip_train=false     # Skip training stages.
skip_eval=false      # Skip decoding and evaluation stages.
skip_upload=true # Skip packing and uploading to zenodo
skip_upload_hf=true  # Skip uploading to hugging face stages.
eval_valid_set=false # Run decoding for the validation set
ngpu=1               # The number of gpus ("0" uses cpu, otherwise use gpu).
num_nodes=1          # The number of nodes.
nj=32                # The number of parallel jobs.
inference_nj=32      # The number of parallel jobs in decoding.
gpu_inference=false  # Whether to perform gpu decoding.
dumpdir=dump         # Directory to dump features.
expdir=exp           # Directory to save experiments.
python=python3       # Specify python to execute espnet commands.

# Data preparation related
local_data_opts= # The options given to local/data.sh.
post_process_local_data_opts= # The options given to local/data.sh for additional processing in stage 4.
auxiliary_data_tags= # the names of training data for auxiliary tasks

# Speed perturbation related
speed_perturb_factors=  # perturbation factors, e.g. "0.9 1.0 1.1" (separated by space).

# Feature extraction related
visual_feats_type=raw       # Feature type (raw, raw_copy, or extracted).
visual_format=mp4    # Visual format: mp4  (only in visual_feats_type=raw).
multi_columns_input_mp4_scp=false  # Enable multi columns mode for input mp4.scp for format_mp4_scp.py
multi_columns_output_mp4_scp=false # Enable multi columns mode for output mp4.scp for format_mp4_scp.py
fps=30
audio_feats_type=raw       # Feature type (raw, raw_copy, fbank_pitch, or extracted).
audio_format=wav    # Audio format: wav, flac, wav.ark, flac.ark  (only in audio_feats_type=raw).
multi_columns_input_wav_scp=false  # Enable multi columns mode for input wav.scp for format_wav_scp.py
multi_columns_output_wav_scp=false # Enable multi columns mode for output wav.scp for format_wav_scp.py
fs=16k               # Sampling rate.
min_wav_duration=0.1 # Minimum duration in second.
max_wav_duration=20  # Maximum duration in second.

# Tokenization related
token_type=bpe      # Tokenization type (char or bpe).
nbpe=30             # The number of BPE vocabulary.
bpemode=unigram     # Mode of BPE (unigram or bpe).
oov="<unk>"         # Out of vocabulary symbol.
blank="<blank>"     # CTC blank symbol
sos_eos="<sos/eos>" # sos and eos symbole
bpe_input_sentence_size=100000000 # Size of input sentence for BPE.
bpe_max_sentence_length=200 # Max length of input sentence for BPE.
bpe_nlsyms=         # non-linguistic symbols list, separated by a comma or a file containing 1 symbol per line, for BPE
bpe_char_cover=1.0  # character coverage when modeling BPE
hugging_face_model_name_or_path="" # Hugging Face model or path for hugging_face tokenizer

# Ngram model related
use_ngram=false
ngram_exp=
ngram_num=3

# Language model related
use_lm=true       # Use language model for AVSR decoding.
lm_tag=           # Suffix to the result dir for language model training.
lm_exp=           # Specify the directory path for LM experiment.
                  # If this option is specified, lm_tag is ignored.
lm_stats_dir=     # Specify the directory path for LM statistics.
lm_config=        # Config for language model training.
lm_args=          # Arguments for language model training, e.g., "--max_epoch 10".
                  # Note that it will overwrite args in lm config.
use_word_lm=false # Whether to use word language model.
num_splits_lm=1   # Number of splitting for lm corpus.
# shellcheck disable=SC2034
word_vocab_size=10000 # Size of word vocabulary.

# AVSR model related
avsr_task=avsr   # AVSR task mode. Either 'avsr' or 'avsr_transducer'.
avsr_tag=       # Suffix to the result dir for avsr model training.
avsr_exp=       # Specify the directory path for AVSR experiment.
               # If this option is specified, avsr_tag is ignored.
avsr_stats_dir= # Specify the directory path for AVSR statistics.
avsr_config=    # Config for avsr model training.
avsr_args=      # Arguments for avsr model training, e.g., "--max_epoch 10".
               # Note that it will overwrite args in avsr config.
pretrained_model=              # Pretrained model to load
ignore_init_mismatch=false      # Ignore initial mismatch
feats_normalize=            # Normalizaton layer type.
num_splits_avsr=1           # Number of splitting for lm corpus.
num_ref=1   # Number of references for training.
            # In supervised learning based speech enhancement / separation, it is equivalent to number of speakers.
num_inf=    # Number of inferences output by the model
            # Note that if it is not specified, it will be the same as num_ref. Otherwise, it will be overwritten.
            # In MixIT, number of outputs is larger than that of references.
sot_avsr=false   # Whether to use Serialized Output Training (SOT)

# Upload model related
hf_repo=

# Decoding related
use_k2=false      # Whether to use k2 based decoder
k2_ctc_decoding=true
use_nbest_rescoring=true # use transformer-decoder
                         # and transformer language model for nbest rescoring
num_paths=1000 # The 3rd argument of k2.random_paths.
nll_batch_size=100 # Affect GPU memory usage when computing nll
                   # during nbest rescoring
k2_config=./conf/decode_avsr_transformer_with_k2.yaml

use_streaming=false # Whether to use streaming decoding

use_maskctc=false # Whether to use maskctc decoding

batch_size=1
inference_tag=    # Suffix to the result dir for decoding.
inference_config= # Config for decoding.
inference_args=   # Arguments for decoding, e.g., "--lm_weight 0.1".
                  # Note that it will overwrite args in inference config.
inference_lm=valid.loss.ave.pth       # Language model path for decoding.
inference_ngram=${ngram_num}gram.bin
inference_avsr_model=valid.acc.ave.pth # AVSR model path for decoding.
                                      # e.g.
                                      # inference_avsr_model=train.loss.best.pth
                                      # inference_avsr_model=3epoch.pth
                                      # inference_avsr_model=valid.acc.best.pth
                                      # inference_avsr_model=valid.loss.ave.pth
download_model= # Download a model from Model Zoo and use it for decoding.

# [Task dependent] Set the datadir name created by local/data.sh
train_set=       # Name of training set.
valid_set=       # Name of validation set used for monitoring/tuning network training.
test_sets=       # Names of test sets. Multiple items (e.g., both dev and eval sets) can be specified.
bpe_train_text=  # Text file path of bpe training set.
lm_train_text=   # Text file path of language model training set.
lm_dev_text=     # Text file path of language model development set.
lm_test_text=    # Text file path of language model evaluation set.
nlsyms_txt=none  # Non-linguistic symbol list if existing.
cleaner=none     # Text cleaner.
hyp_cleaner=none # Text cleaner for hypotheses (may be used with external tokenizers)
g2p=none         # g2p method (needed if token_type=phn).
lang=noinfo      # The language type of corpus.
score_opts=                # The options given to sclite scoring
local_score_opts=          # The options given to local/score.sh.
avsr_speech_fold_length=800 # fold_length for speech data during AVSR training.
avsr_text_fold_length=150   # fold_length for text data during AVSR training.
lm_fold_length=150         # fold_length for LM training.

help_message=$(cat << EOF
Usage: $0 --train-set "<train_set_name>" --valid-set "<valid_set_name>" --test_sets "<test_set_names>"

Options:
    # General configuration
    --stage          # Processes starts from the specified stage (default="${stage}").
    --stop_stage     # Processes is stopped at the specified stage (default="${stop_stage}").
    --skip_stages    # Spicify the stage to be skipped (default="${skip_stages}").
    --skip_data_prep # Skip data preparation stages (default="${skip_data_prep}").
    --skip_train     # Skip training stages (default="${skip_train}").
    --skip_eval      # Skip decoding and evaluation stages (default="${skip_eval}").
    --skip_upload    # Skip packing and uploading stages (default="${skip_upload}").
    --skip_upload_hf    # Skip packing and uploading stages (default="${skip_upload_hf}").
    --eval_valid_set # Run decoding for the validation set (default="${eval_valid_set}").
    --ngpu           # The number of gpus ("0" uses cpu, otherwise use gpu, default="${ngpu}").
    --num_nodes      # The number of nodes (default="${num_nodes}").
    --nj             # The number of parallel jobs (default="${nj}").
    --inference_nj   # The number of parallel jobs in decoding (default="${inference_nj}").
    --gpu_inference  # Whether to perform gpu decoding (default="${gpu_inference}").
    --dumpdir        # Directory to dump features (default="${dumpdir}").
    --expdir         # Directory to save experiments (default="${expdir}").
    --python         # Specify python to execute espnet commands (default="${python}").

    # Data preparation related
    --local_data_opts # The options given to local/data.sh (default="${local_data_opts}").

    # Speed perturbation related
    --speed_perturb_factors # speed perturbation factors, e.g. "0.9 1.0 1.1" (separated by space, default="${speed_perturb_factors}").

    # Feature extraction related
    --visual_feats_type       # Feature type (raw, raw_copy, or extracted, default="${visual_feats_type}").
    --visual_format     # Visual format: mp4  (only in visual_feats_type=raw or raw_copy, default="${visual_format}").
    --fps               # Frames per second (default="${fps}").
    --audio_feats_type       # Feature type (raw, raw_copy, fbank_pitch or extracted, default="${audio_feats_type}").
    --audio_format     # Audio format: wav, flac, wav.ark, flac.ark  (only in audio_feats_type=raw or raw_copy, default="${audio_format}").
    --fs               # Sampling rate (default="${fs}").
    --min_wav_duration # Minimum duration in second (default="${min_wav_duration}").
    --max_wav_duration # Maximum duration in second (default="${max_wav_duration}").

    # Tokenization related
    --token_type              # Tokenization type (char or bpe, default="${token_type}").
    --nbpe                    # The number of BPE vocabulary (default="${nbpe}").
    --bpemode                 # Mode of BPE (unigram or bpe, default="${bpemode}").
    --oov                     # Out of vocabulary symbol (default="${oov}").
    --blank                   # CTC blank symbol (default="${blank}").
    --sos_eos                 # sos and eos symbole (default="${sos_eos}").
    --bpe_input_sentence_size # Size of input sentence for BPE (default="${bpe_input_sentence_size}").
    --bpe_max_sentence_length # Max Length of input sentence for BPE
    --bpe_nlsyms              # Non-linguistic symbol list for sentencepiece, separated by a comma or a file containing 1 symbol per line . (default="${bpe_nlsyms}").
    --bpe_char_cover          # Character coverage when modeling BPE (default="${bpe_char_cover}").

    # Language model related
    --lm_tag          # Suffix to the result dir for language model training (default="${lm_tag}").
    --lm_exp          # Specify the directory path for LM experiment.
                      # If this option is specified, lm_tag is ignored (default="${lm_exp}").
    --lm_stats_dir    # Specify the directory path for LM statistics (default="${lm_stats_dir}").
    --lm_config       # Config for language model training (default="${lm_config}").
    --lm_args         # Arguments for language model training (default="${lm_args}").
                      # e.g., --lm_args "--max_epoch 10"
                      # Note that it will overwrite args in lm config.
    --use_word_lm     # Whether to use word language model (default="${use_word_lm}").
    --word_vocab_size # Size of word vocabulary (default="${word_vocab_size}").
    --num_splits_lm   # Number of splitting for lm corpus (default="${num_splits_lm}").

    # AVSR model related
    --avsr_task         # AVSR task mode. Either 'avsr' or 'avsr_transducer'. (default="${avsr_task}").
    --avsr_tag          # Suffix to the result dir for avsr model training (default="${avsr_tag}").
    --avsr_exp          # Specify the directory path for AVSR experiment.
                       # If this option is specified, avsr_tag is ignored (default="${avsr_exp}").
    --avsr_stats_dir    # Specify the directory path for AVSR statistics (default="${avsr_stats_dir}").
    --avsr_config       # Config for avsr model training (default="${avsr_config}").
    --avsr_args         # Arguments for avsr model training (default="${avsr_args}").
                       # e.g., --avsr_args "--max_epoch 10"
                       # Note that it will overwrite args in avsr config.
    --pretrained_model=          # Pretrained model to load (default="${pretrained_model}").
    --ignore_init_mismatch=      # Ignore mismatch parameter init with pretrained model (default="${ignore_init_mismatch}").
    --feats_normalize  # Normalizaton layer type (default="${feats_normalize}").
    --num_splits_avsr   # Number of splitting for lm corpus  (default="${num_splits_avsr}").
    --num_ref    # Number of references for training (default="${num_ref}").
                 # In supervised learning based speech recognition, it is equivalent to number of speakers.
    --num_inf    # Number of inference audio generated by the model (default="${num_inf}")
                 # Note that if it is not specified, it will be the same as num_ref. Otherwise, it will be overwritten.
    --sot_avsr    # Whether to use Serialized Output Training (SOT) (default="${sot_avsr}")

    # Decoding related
    --inference_tag       # Suffix to the result dir for decoding (default="${inference_tag}").
    --inference_config    # Config for decoding (default="${inference_config}").
    --inference_args      # Arguments for decoding (default="${inference_args}").
                          # e.g., --inference_args "--lm_weight 0.1"
                          # Note that it will overwrite args in inference config.
    --inference_lm        # Language model path for decoding (default="${inference_lm}").
    --inference_avsr_model # AVSR model path for decoding (default="${inference_avsr_model}").
    --download_model      # Download a model from Model Zoo and use it for decoding (default="${download_model}").
    --use_streaming       # Whether to use streaming decoding (default="${use_streaming}").
    --use_maskctc         # Whether to use maskctc decoding (default="${use_streaming}").

    # [Task dependent] Set the datadir name created by local/data.sh
    --train_set     # Name of training set (required).
    --valid_set     # Name of validation set used for monitoring/tuning network training (required).
    --test_sets     # Names of test sets.
                    # Multiple items (e.g., both dev and eval sets) can be specified (required).
    --bpe_train_text # Text file path of bpe training set.
    --lm_train_text  # Text file path of language model training set.
    --lm_dev_text   # Text file path of language model development set (default="${lm_dev_text}").
    --lm_test_text  # Text file path of language model evaluation set (default="${lm_test_text}").
    --nlsyms_txt    # Non-linguistic symbol list if existing (default="${nlsyms_txt}").
    --cleaner       # Text cleaner (default="${cleaner}").
    --g2p           # g2p method (default="${g2p}").
    --lang          # The language type of corpus (default=${lang}).
    --score_opts             # The options given to sclite scoring (default="{score_opts}").
    --local_score_opts       # The options given to local/score.sh (default="{local_score_opts}").
    --avsr_speech_fold_length # fold_length for speech data during AVSR training (default="${avsr_speech_fold_length}").
    --avsr_text_fold_length   # fold_length for text data during AVSR training (default="${avsr_text_fold_length}").
    --lm_fold_length         # fold_length for LM training (default="${lm_fold_length}").
EOF
)

log "$0 $*"
# Save command line args for logging (they will be lost after utils/parse_options.sh)
run_args=$(scripts/utils/print_args.sh $0 "$@")
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "${help_message}"
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh
. ./cmd.sh


# Check required arguments
if ! "${skip_train}"; then
    [ -z "${train_set}" ] && { log "${help_message}"; log "Error: --train_set is required"; exit 2; };
    [ -z "${valid_set}" ] && { log "${help_message}"; log "Error: --valid_set is required"; exit 2; };
fi
if ! "${eval_valid_set}"; then
    [ -z "${test_sets}" ] && { log "${help_message}"; log "Error: --test_sets is required"; exit 2; };
else
    [ -z "${valid_set}" ] && { log "${help_message}"; log "Error: --valid_set is required"; exit 2; };
fi

if [ -n "${train_set}" ] && [ "${train_set}" = "${valid_set}" ]; then
    log "Error: train_set and valid_set must be different. --train_set ${train_set} --valid_set ${valid_set}"
    exit 1
fi

_test_sets=
for dset in ${test_sets}; do
    if [ "${dset}" = "${train_set}" ]; then
        log "Error: train_set and test_sets must be different. --train_set ${train_set} --test_sets ${test_sets}"
        exit 1
    fi
    if [ "${dset}" = "${valid_set}" ]; then
        log "Info: The valid_set '${valid_set}' is included in the test_sets. '--eval_valid_set true' is set and '${valid_set}' is removed from the test_sets"
        eval_valid_set=true
    elif [[ " ${_test_sets} " =~ [[:space:]]${dset}[[:space:]] ]]; then
        log "Info: ${dset} is duplicated in the test_sets. One is removed"
    else
        _test_sets+="${dset} "
    fi
done
test_sets=${_test_sets}

# Check feature type
if [ "${audio_feats_type}" = raw ]; then
    audio_data_feats=${dumpdir}/raw
elif [ "${audio_feats_type}" = raw_copy ]; then
    # raw_copy is as same as raw except for skipping the format_wav stage
    audio_data_feats=${dumpdir}/raw_copy
elif [ "${audio_feats_type}" = fbank_pitch ]; then
    audio_data_feats=${dumpdir}/fbank_pitch
elif [ "${audio_feats_type}" = fbank ]; then
    audio_data_feats=${dumpdir}/fbank
elif [ "${audio_feats_type}" == extracted ]; then
    audio_data_feats=${dumpdir}/extracted
else
    log "${help_message}"
    log "Error: not supported: --audio_feats_type ${audio_feats_type}"
    exit 2
fi

if [ "${visual_feats_type}" = raw ]; then
    visual_data_feats=${dumpdir}/raw
elif [ "${visual_feats_type}" = raw_copy ]; then
    # raw_copy is as same as raw except for skipping the format_wav stage
    visual_data_feats=${dumpdir}/raw_copy
elif [ "${visual_feats_type}" == extracted ]; then
    visual_data_feats=${dumpdir}/extracted
else
    log "${help_message}"
    log "Error: not supported: --visual_feats_type ${visual_feats_type}"
    exit 2
fi


num_inf=${num_inf:=${num_ref}}
# Preprocessor related
if [ ${num_ref} -eq 1 ]; then
    # For single speaker, text file path and name are text
    ref_text_files_str="text "
    ref_text_names_str="text "
else
    # For multiple speakers, text file path and name are text_spk[1-N] and [text, text_spk2, ...]
    #TODO(simpleoier): later to support flexibly defined text prefix
    ref_text_files_str="text_spk1 "
    ref_text_names_str="text "
    for n in $(seq 2 ${num_ref}); do
        ref_text_files_str+="text_spk${n} "
        ref_text_names_str+="text_spk${n} "
    done
fi
# shellcheck disable=SC2206
ref_text_files=(${ref_text_files_str// / })
# shellcheck disable=SC2206
ref_text_names=(${ref_text_names_str// / })

[ -z "${bpe_train_text}" ] && bpe_train_text="${audio_data_feats}/org/${train_set}/${ref_text_files[0]}"
# Use the same text as AVSR for lm training if not specified.
[ -z "${lm_train_text}" ] && lm_train_text="${audio_data_feats}/org/${train_set}/${ref_text_files[0]}"
# Use the same text as AVSR for lm training if not specified.
[ -z "${lm_dev_text}" ] && lm_dev_text="${audio_data_feats}/org/${valid_set}/${ref_text_files[0]}"
if [ -z "${lm_test_text}" ]; then
    if [ -z "${test_sets}" ]; then
        lm_test_text="${audio_data_feats}/org/${valid_set}/${ref_text_files[0]}"
    else
        # Use the text of the 1st evaldir if lm_test is not specified
        lm_test_text="${audio_data_feats}/${test_sets%% *}/${ref_text_files[0]}"
    fi
fi

# Check tokenization type
if [ "${lang}" != noinfo ]; then
    token_listdir=data/${lang}_token_list
else
    token_listdir=data/token_list
fi
bpedir="${token_listdir}/bpe_${bpemode}${nbpe}"
bpeprefix="${bpedir}"/bpe
bpemodel="${bpeprefix}".model
bpetoken_list="${bpedir}"/tokens.txt
chartoken_list="${token_listdir}"/char/tokens.txt
hugging_face_token_list="${token_listdir}/hugging_face_"${hugging_face_model_name_or_path/\//-}/tokens.txt
# NOTE: keep for future development.
# shellcheck disable=SC2034
wordtoken_list="${token_listdir}"/word/tokens.txt

if [ "${token_type}" = bpe ]; then
    token_list="${bpetoken_list}"
elif [ "${token_type}" = char ]; then
    token_list="${chartoken_list}"
    bpemodel=none
elif [ "${token_type}" = word ]; then
    token_list="${wordtoken_list}"
    bpemodel=none
elif [ "${token_type}" = whisper_en ]; then # should make token_list an output filepath here
    token_list="${token_listdir}"/whisper_en/tokens.txt
    bpemodel=whisper_en
    hyp_cleaner=${cleaner}
elif [ "${token_type}" = whisper_multilingual ]; then
    token_list="${token_listdir}"/whisper_multilingual/tokens.txt
    bpemodel=whisper_multilingual
    hyp_cleaner=${cleaner}
elif [ "${token_type}" = hugging_face ]; then
    token_list="${hugging_face_token_list}"
    bpemodel=${hugging_face_model_name_or_path}
else
    log "Error: not supported --token_type '${token_type}'"
    exit 2
fi
if ${use_word_lm}; then
    log "Error: Word LM is not supported yet"
    exit 2
else
    lm_token_list="${token_list}"
    lm_token_type="${token_type}"
fi


# Set tag for naming of model directory
if [ -z "${avsr_tag}" ]; then
    if [ -n "${avsr_config}" ]; then
        avsr_tag="$(basename "${avsr_config}" .yaml)_${audio_feats_type}_${visual_feats_type}"
    else
        avsr_tag="train_${audio_feats_type}_${visual_feats_type}"
    fi
    if [ "${lang}" != noinfo ]; then
        avsr_tag+="_${lang}_${token_type}"
    else
        avsr_tag+="_${token_type}"
    fi
    if [ "${token_type}" = bpe ]; then
        avsr_tag+="${nbpe}"
    fi
    if [ "${token_type}" = hugging_face ]; then
        avsr_tag+="_"${hugging_face_model_name_or_path/\//-}
    fi
    # Add overwritten arg's info
    if [ -n "${avsr_args}" ]; then
        avsr_tag+="$(echo "${avsr_args}" | sed -e "s/--/\_/g" -e "s/[ |=/]//g")"
    fi
    if [ -n "${speed_perturb_factors}" ]; then
        avsr_tag+="_sp"
    fi
fi
if [ -z "${lm_tag}" ]; then
    if [ -n "${lm_config}" ]; then
        lm_tag="$(basename "${lm_config}" .yaml)"
    else
        lm_tag="train"
    fi
    if [ "${lang}" != noinfo ]; then
        lm_tag+="_${lang}_${lm_token_type}"
    else
        lm_tag+="_${lm_token_type}"
    fi
    if [ "${lm_token_type}" = bpe ]; then
        lm_tag+="${nbpe}"
    fi
    # Add overwritten arg's info
    if [ -n "${lm_args}" ]; then
        lm_tag+="$(echo "${lm_args}" | sed -e "s/--/\_/g" -e "s/[ |=/]//g")"
    fi
fi

# The directory used for collect-stats mode
if [ -z "${avsr_stats_dir}" ]; then
    if [ "${lang}" != noinfo ]; then
        avsr_stats_dir="${expdir}/avsr_stats_${audio_feats_type}_${visual_feats_type}_${lang}_${token_type}"
    else
        avsr_stats_dir="${expdir}/avsr_stats_${audio_feats_type}_${visual_feats_type}_${token_type}"
    fi
    if [ "${token_type}" = bpe ]; then
        avsr_stats_dir+="${nbpe}"
    fi
    if [ "${token_type}" = hugging_face ]; then
        avsr_stats_dir+="_"${hugging_face_model_name_or_path/\//-}
    fi
    if [ -n "${speed_perturb_factors}" ]; then
        avsr_stats_dir+="_sp"
    fi
fi
if [ -z "${lm_stats_dir}" ]; then
    if [ "${lang}" != noinfo ]; then
        lm_stats_dir="${expdir}/lm_stats_${lang}_${lm_token_type}"
    else
        lm_stats_dir="${expdir}/lm_stats_${lm_token_type}"
    fi
    if [ "${lm_token_type}" = bpe ]; then
        lm_stats_dir+="${nbpe}"
    fi
fi
# The directory used for training commands
if [ -z "${avsr_exp}" ]; then
    avsr_exp="${expdir}/avsr_${avsr_tag}"
fi
if [ -z "${lm_exp}" ]; then
    lm_exp="${expdir}/lm_${lm_tag}"
fi
if [ -z "${ngram_exp}" ]; then
    ngram_exp="${expdir}/ngram"
fi


if [ -z "${inference_tag}" ]; then
    if [ -n "${inference_config}" ]; then
        inference_tag="$(basename "${inference_config}" .yaml)"
    else
        inference_tag=inference
    fi
    # Add overwritten arg's info
    if [ -n "${inference_args}" ]; then
        inference_tag+="$(echo "${inference_args}" | sed -e "s/--/\_/g" -e "s/[ |=]//g")"
    fi
    if "${use_lm}"; then
        inference_tag+="_lm_$(basename "${lm_exp}")_$(echo "${inference_lm}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"
    fi
    if "${use_ngram}"; then
        inference_tag+="_ngram_$(basename "${ngram_exp}")_$(echo "${inference_ngram}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"
    fi
    inference_tag+="_avsr_model_$(echo "${inference_avsr_model}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"

    if "${use_k2}"; then
      inference_tag+="_use_k2"
      inference_tag+="_k2_ctc_decoding_${k2_ctc_decoding}"
      inference_tag+="_use_nbest_rescoring_${use_nbest_rescoring}"
    fi
fi

if "${skip_data_prep}"; then
    skip_stages+="1 2 3 4 5 "
fi
if "${skip_train}"; then
    skip_stages+="2 4 5 6 7 8 9 10 11 "
elif ! "${use_lm}"; then
    skip_stages+="6 7 8 "
fi
if ! "${use_ngram}"; then
    skip_stages+="9 "
fi
if "${skip_eval}"; then
    skip_stages+="12 13 "
fi
if [ -n "${download_model}" ]; then
    skip_stages+="14 "
fi
if "${skip_upload}"; then
    skip_stages+="14 15 "
fi
if "${skip_upload_hf}"; then
    skip_stages+="14 16 "
fi
skip_stages=$(echo "${skip_stages}" | tr ' ' '\n' | sort -nu | tr '\n' ' ')
log "Skipped stages: ${skip_stages}"

# ========================== Main stages start from here. ==========================



if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ] && ! [[ " ${skip_stages} " =~ [[:space:]]1[[:space:]] ]]; then
    log "Stage 1: Data preparation for data/${train_set}, data/${valid_set}, etc."
    # [Task dependent] Need to create data.sh for new corpus
    local/data.sh ${local_data_opts}
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ] && ! [[ " ${skip_stages} " =~ [[:space:]]2[[:space:]] ]]; then
    if [ -n "${speed_perturb_factors}" ]; then
        # NOTE(JK): Visual perturbation to be updated, to synchronize with visual data.
        log "[Error] ``Stage 2: Speed perturbation`` is to be updated."
        exit 1

        log "Stage 2: Speed perturbation: data/${train_set} -> data/${train_set}_sp"
        for factor in ${speed_perturb_factors}; do
           if python3 -c "assert ${factor} != 1.0" 2>/dev/null; then
               scripts/utils/perturb_data_dir_speed.sh \
                   ${ref_text_files_str:+--utt_extra_files "${ref_text_files_str}"} \
                   "${factor}" "data/${train_set}" "data/${train_set}_sp${factor}"
               _dirs+="data/${train_set}_sp${factor} "
           else
               # If speed factor is 1, same as the original
               _dirs+="data/${train_set} "
           fi
        done
        utils/combine_data.sh \
            ${ref_text_files_str:+--extra_files "${ref_text_files_str}"} \
            "data/${train_set}_sp" ${_dirs}
    else
       log "Skip stage 2: Speed perturbation"
    fi
fi

if [ -n "${speed_perturb_factors}" ]; then
    # NOTE(JK): Visual perturbation to be updated, to synchronize with visual data.
    log "[Error] ``Stage 2: Speed perturbation`` is to be updated."
    exit 1

    train_set="${train_set}_sp"
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ] && ! [[ " ${skip_stages} " =~ [[:space:]]3[[:space:]] ]]; then
    if "${skip_train}"; then
        if "${eval_valid_set}"; then
            _dsets="${valid_set} ${test_sets}"
        else
            _dsets="${test_sets}"
        fi
    else
        _dsets="${train_set} ${valid_set} ${test_sets}"
    fi
    if [ "${audio_feats_type}" = raw ]; then
        log "Stage 3: Format wav.scp: data/ -> ${audio_data_feats}"

        # ====== Recreating "wav.scp" ======
        # Kaldi-wav.scp, which can describe the file path with unix-pipe, like "cat /some/path |",
        # shouldn't be used in training process.
        # "format_wav_scp.sh" dumps such pipe-style-wav to real audio file
        # and it can also change the audio-format and sampling rate.
        # If nothing is need, then format_wav_scp.sh does nothing:
        # i.e. the input file format and rate is same as the output.

        for dset in ${_dsets}; do
            if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
                _suf="/org"
            else
                _suf=""
            fi
            utils/copy_data_dir.sh --validate_opts --non-print data/"${dset}" "${audio_data_feats}${_suf}/${dset}"
            rm -f ${audio_data_feats}${_suf}/${dset}/{segments,wav.scp,reco2file_and_channel,reco2dur}

            # Copy reference text files if there is more than 1 reference
            if [ ${#ref_text_files[@]} -gt 1 ]; then
                # shellcheck disable=SC2068
                for ref_txt in ${ref_text_files[@]}; do
                    [ -f data/${dset}/${ref_txt} ] && cp data/${dset}/${ref_txt} ${audio_data_feats}${_suf}/${dset}
                done
            fi

            _opts=
            if [ -e data/"${dset}"/segments ]; then
                # "segments" is used for splitting wav files which are written in "wav".scp
                # into utterances. The file format of segments:
                #   <segment_id> <record_id> <start_time> <end_time>
                #   "e.g. call-861225-A-0050-0065 call-861225-A 5.0 6.5"
                # Where the time is written in seconds.
                _opts+="--segments data/${dset}/segments "
            fi
            # shellcheck disable=SC2086
            scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
                --audio-format "${audio_format}" --fs "${fs}" ${_opts} \
                --multi-columns-input "${multi_columns_input_wav_scp}" \
                --multi-columns-output "${multi_columns_output_wav_scp}" \
                "data/${dset}/wav.scp" "${audio_data_feats}${_suf}/${dset}"

            echo "${audio_feats_type}" > "${audio_data_feats}${_suf}/${dset}/audio_feats_type"
            if "${multi_columns_output_wav_scp}"; then
                echo "multi_${audio_format}" > "${audio_data_feats}${_suf}/${dset}/audio_format"
            else
                echo "${audio_format}" > "${audio_data_feats}${_suf}/${dset}/audio_format"
            fi
        done

    elif [ "${audio_feats_type}" = raw_copy ]; then
        # If you guaranteed that the data already satisfy the raw format, you can skip format_wav_scp.py for reduce the overhead
        for dset in ${_dsets}; do
            if [ -e "data/${dset}/segments" ]; then
                log "Error: data/${dset}/segments is existing. Please use --audio_feats_type raw"
                exit 1
            fi
            if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
                _suf="/org"
            else
                _suf=""
            fi
            utils/copy_data_dir.sh --validate_opts --non-print data/"${dset}" "${audio_data_feats}${_suf}/${dset}"
            if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
                _suf="/org"

                if [ -e "data/${dset}/utt2dur" ]; then
                    _fs=$(python3 -c "import humanfriendly as h;print(h.parse_size('${fs}'))")
                    <data/${dset}/utt2dur awk '{ print $1, int($2*'${_fs}'); }' > "${audio_data_feats}${_suf}/${dset}"/utt2num_samples

                elif [ -e "data/${dset}/utt2num_samples" ]; then
                    cp "data/${dset}/utt2num_samples" "${audio_data_feats}${_suf}/${dset}"/utt2num_samples

                else
                    log "Error: data/${dset}/utt2dur or data/${dset}/utt2num_samples must be existing for train_set and valid_set. Please use --audio_feats_type raw. If you'd like to perform this script for evaluation, please give --skip_train true"
                    exit 1
                fi
            fi

            # Copy reference text files if there is more than 1 reference
            if [ ${#ref_text_files[@]} -gt 1 ]; then
                # shellcheck disable=SC2068
                for ref_txt in ${ref_text_files[@]}; do
                    [ -f data/${dset}/${ref_txt} ] && cp data/${dset}/${ref_txt} ${audio_data_feats}${_suf}/${dset}
                done
            fi

            echo "raw" > "${audio_data_feats}${_suf}/${dset}/audio_feats_type"
            if "${multi_columns_input_wav_scp}"; then
                echo "multi_${audio_format}" > "${audio_data_feats}${_suf}/${dset}/audio_format"
            else
                echo "${audio_format}" > "${audio_data_feats}${_suf}/${dset}/audio_format"
            fi
        done

    elif [ "${audio_feats_type}" = fbank_pitch ]; then
        log "[Require Kaldi] Stage 3: ${audio_feats_type} extract: data/ -> ${audio_data_feats}"

        for dset in ${_dsets}; do
            if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
                _suf="/org"
            else
                _suf=""
            fi
            # 1. Copy datadir
            utils/copy_data_dir.sh --validate_opts --non-print data/"${dset}" "${audio_data_feats}${_suf}/${dset}"

            # Copy reference text files if there is more than 1 reference
            if [ ${#ref_text_files[@]} -gt 1 ]; then
                # shellcheck disable=SC2068
                for ref_txt in ${ref_text_files[@]}; do
                    [ -f data/${dset}/${ref_txt} ] && cp data/${dset}/${ref_txt} ${audio_data_feats}${_suf}/${dset}
                done
            fi

            # 2. Feature extract
            _nj=$(min "${nj}" "$(<"${audio_data_feats}${_suf}/${dset}/utt2spk" wc -l)")
            steps/make_fbank_pitch.sh --nj "${_nj}" --cmd "${train_cmd}" "${audio_data_feats}${_suf}/${dset}"
            utils/fix_data_dir.sh "${audio_data_feats}${_suf}/${dset}"

            # 3. Derive the the frame length and feature dimension
            scripts/feats/feat_to_shape.sh --nj "${_nj}" --cmd "${train_cmd}" \
                "${audio_data_feats}${_suf}/${dset}/feats.scp" "${audio_data_feats}${_suf}/${dset}/feats_shape"

            # 4. Write feats_dim
            head -n 1 "${audio_data_feats}${_suf}/${dset}/feats_shape" | awk '{ print $2 }' \
                | cut -d, -f2 > ${audio_data_feats}${_suf}/${dset}/feats_dim

            # 5. Write audio_feats_type
            echo "${audio_feats_type}" > "${audio_data_feats}${_suf}/${dset}/audio_feats_type"
        done

    elif [ "${audio_feats_type}" = fbank ]; then
        log "Stage 3: ${audio_feats_type} extract: data/ -> ${audio_data_feats}"
        log "${audio_feats_type} is not supported yet."
        exit 1

    elif  [ "${audio_feats_type}" = extracted ]; then
        log "Stage 3: ${audio_feats_type} extract: data/ -> ${audio_data_feats}"
        # Assumming you don't have wav.scp, but feats.scp is created by local/data.sh instead.

        for dset in ${_dsets}; do
            if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
                _suf="/org"
            else
                _suf=""
            fi
            # Generate dummy wav.scp to avoid error by copy_data_dir.sh
            if [ ! -f data/"${dset}"/wav.scp ]; then
                if [ ! -f data/"${dset}"/segments ]; then
                    <data/"${dset}"/feats.scp awk ' { print($1,"<DUMMY>") }' > data/"${dset}"/wav.scp
                else
                    <data/"${dset}"/segments awk ' { print($2,"<DUMMY>") }' > data/"${dset}"/wav.scp
                fi
            fi
            utils/copy_data_dir.sh --validate_opts --non-print data/"${dset}" "${audio_data_feats}${_suf}/${dset}"

            # Copy reference text files if there is more than 1 reference
            # shellcheck disable=SC2068
            if [ ${#ref_text_files[@]} -gt 1 ]; then
                for ref_txt in ${ref_text_files[@]}; do
                    [ -f data/${dset}/${ref_txt} ] && cp data/${dset}/${ref_txt} ${audio_data_feats}${_suf}/${dset}
                done
            fi

            # Derive the the frame length and feature dimension
            _nj=$(min "${nj}" "$(<"${audio_data_feats}${_suf}/${dset}/utt2spk" wc -l)")
            scripts/feats/feat_to_shape.sh --nj "${_nj}" --cmd "${train_cmd}" \
                "${audio_data_feats}${_suf}/${dset}/feats.scp" "${audio_data_feats}${_suf}/${dset}/feats_shape"

            pyscripts/feats/feat-to-shape.py "scp:head -n 1 ${audio_data_feats}${_suf}/${dset}/feats.scp |" - | \
                awk '{ print $2 }' | cut -d, -f2 > "${audio_data_feats}${_suf}/${dset}/feats_dim"

            echo "${audio_feats_type}" > "${audio_data_feats}${_suf}/${dset}/audio_feats_type"
        done

    else
        log "Error: not supported: --audio_feats_type ${audio_feats_type}"
        exit 2
    fi

    # NOTE(JK): processing for ${visual_feats_type} to be updated
    for dset in ${_dsets}; do
        if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
            _suf="/org"
        else
            _suf=""
        fi
        echo "${visual_feats_type}" > "${visual_data_feats}${_suf}/${dset}/visual_feats_type"
        echo "${visual_format}" > "${visual_data_feats}${_suf}/${dset}/visual_format"
    done
fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ] && ! [[ " ${skip_stages} " =~ [[:space:]]4[[:space:]] ]]; then
    log "Stage 4: Remove long/short data: ${audio_data_feats}/org -> ${audio_data_feats}"

    # NOTE(kamo): Not applying to test_sets to keep original data
    for dset in "${train_set}" "${valid_set}"; do

        # Copy data dir
        utils/copy_data_dir.sh --validate_opts --non-print "${audio_data_feats}/org/${dset}" "${audio_data_feats}/${dset}"
        cp "${audio_data_feats}/org/${dset}/audio_feats_type" "${audio_data_feats}/${dset}/audio_feats_type"
        cp "${visual_data_feats}/org/${dset}/visual_feats_type" "${visual_data_feats}/${dset}/visual_feats_type"

        # Remove short utterances
        _audio_feats_type="$(<${audio_data_feats}/${dset}/audio_feats_type)"
        if [ "${_audio_feats_type}" = raw ]; then
            _fs=$(python3 -c "import humanfriendly as h;print(h.parse_size('${fs}'))")
            _min_length=$(python3 -c "print(int(${min_wav_duration} * ${_fs}))")
            _max_length=$(python3 -c "print(int(${max_wav_duration} * ${_fs}))")

            # utt2num_samples is created by format_wav_scp.sh
            <"${audio_data_feats}/org/${dset}/utt2num_samples" \
                awk -v min_length="${_min_length}" -v max_length="${_max_length}" \
                    '{ if ($2 > min_length && $2 < max_length ) print $0; }' \
                    >"${audio_data_feats}/${dset}/utt2num_samples"
            <"${audio_data_feats}/org/${dset}/wav.scp" \
                utils/filter_scp.pl "${audio_data_feats}/${dset}/utt2num_samples"  \
                >"${audio_data_feats}/${dset}/wav.scp"
        else
            # Get frame shift in ms from conf/fbank.conf
            _frame_shift=
            if [ -f conf/fbank.conf ] && [ "$(<conf/fbank.conf grep -c frame-shift)" -gt 0 ]; then
                # Assume using conf/fbank.conf for feature extraction
                _frame_shift="$(<conf/fbank.conf grep frame-shift | sed -e 's/[-a-z =]*\([0-9]*\)/\1/g')"
            fi
            if [ -z "${_frame_shift}" ]; then
                # If not existing, use the default number in Kaldi (=10ms).
                # If you are using different number, you have to change the following value manually.
                _frame_shift=10
            fi

            _min_length=$(python3 -c "print(int(${min_wav_duration} / ${_frame_shift} * 1000))")
            _max_length=$(python3 -c "print(int(${max_wav_duration} / ${_frame_shift} * 1000))")

            cp "${audio_data_feats}/org/${dset}/feats_dim" "${audio_data_feats}/${dset}/feats_dim"
            <"${audio_data_feats}/org/${dset}/feats_shape" awk -F, ' { print $1 } ' \
                | awk -v min_length="${_min_length}" -v max_length="${_max_length}" \
                    '{ if ($2 > min_length && $2 < max_length) print $0; }' \
                    >"${audio_data_feats}/${dset}/feats_shape"
            <"${audio_data_feats}/org/${dset}/feats.scp" \
                utils/filter_scp.pl "${audio_data_feats}/${dset}/feats_shape"  \
                >"${audio_data_feats}/${dset}/feats.scp"
        fi

        # Remove empty text
        # shellcheck disable=SC2068
        for ref_txt in ${ref_text_files[@]}; do
            <"${audio_data_feats}/org/${dset}/${ref_txt}" \
                awk ' { if( NF != 1 ) print $0; } ' >"${audio_data_feats}/${dset}/${ref_txt}"
        done

        # fix_data_dir.sh leaves only utts which exist in all files
        utils/fix_data_dir.sh \
            ${ref_text_files_str:+--utt_extra_files "${ref_text_files_str}"} \
            "${audio_data_feats}/${dset}"
    done

    if [ -n "${post_process_local_data_opts}" ]; then
        # Do any additional local data post-processing here
        local/data.sh ${post_process_local_data_opts} --avsr_data_dir "${audio_data_feats}/${train_set}"
    fi

    # shellcheck disable=SC2002,SC2068,SC2005
    for lm_txt in ${lm_train_text[@]}; do
        suffix=$(echo "$(basename ${lm_txt})" | sed 's/text//')
        <${lm_txt} awk -v suffix=${suffix} ' { if( NF != 1 ) {$1=$1 suffix; print $0; }} '
    done > "${audio_data_feats}/lm_train.txt"
fi


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ] && ! [[ " ${skip_stages} " =~ [[:space:]]5[[:space:]] ]]; then
    if [ "${token_type}" = bpe ]; then
        log "Stage 5: Generate token_list from ${bpe_train_text} using BPE"

        mkdir -p "${bpedir}"
        # shellcheck disable=SC2002
        cat ${bpe_train_text} | cut -f 2- -d" "  > "${bpedir}"/train.txt

        if [ -n "${bpe_nlsyms}" ]; then
            if test -f "${bpe_nlsyms}"; then
                bpe_nlsyms_list=$(awk '{print $1}' ${bpe_nlsyms} | paste -s -d, -)
                _opts_spm="--user_defined_symbols=${bpe_nlsyms_list}"
            else
                _opts_spm="--user_defined_symbols=${bpe_nlsyms}"
            fi
        else
            _opts_spm=""
        fi

        if ${sot_avsr}; then
            # For SOT training, we add <sc> as an user-defined modeling unit.
            # The input text may be `text^1 <sc> text^2 <sc> text^3`, where `text^n`
            # refers to the transcription of `speaker n`.
            # The order of different texts is determined by their start times.
            _opts_spm+=" --user_defined_symbols=<sc>"
        fi

        spm_train \
            --input="${bpedir}"/train.txt \
            --vocab_size="${nbpe}" \
            --model_type="${bpemode}" \
            --model_prefix="${bpeprefix}" \
            --character_coverage=${bpe_char_cover} \
            --input_sentence_size="${bpe_input_sentence_size}" \
            --max_sentence_length="${bpe_max_sentence_length}" \
            ${_opts_spm}

        {
        echo "${blank}"
        echo "${oov}"
        # Remove <unk>, <s>, </s> from the vocabulary
        <"${bpeprefix}".vocab awk '{ if( NR != 1 && NR != 2 && NR != 3 ){ print $1; } }'
        echo "${sos_eos}"
        } > "${token_list}"

    elif [ "${token_type}" = char ] || [ "${token_type}" = word ]; then
        log "Stage 5: Generate character level token_list from ${lm_train_text}"

        _opts="--non_linguistic_symbols ${nlsyms_txt}"

        if ${sot_avsr} && [ "${token_type}" = char ]; then
            # For SOT training, we add <sc> as an user-defined modeling unit.
            # The input text may be `text^1 <sc> text^2 <sc> text^3`, where `text^n`
            # refers to the transcription of `speaker n`.
            # The order of different texts is determined by their start times.
            _opts+=" --add_nonsplit_symbol <sc>:2 "
        fi

        # The first symbol in token_list must be "<blank>" and the last must be also sos/eos:
        # 0 is reserved for CTC-blank for AVSR and also used as ignore-index in the other task
        ${python} -m espnet2.bin.tokenize_text  \
            --token_type "${token_type}" \
            --input "${audio_data_feats}/lm_train.txt" --output "${token_list}" ${_opts} \
            --field 2- \
            --cleaner "${cleaner}" \
            --g2p "${g2p}" \
            --write_vocabulary true \
            --add_symbol "${blank}:0" \
            --add_symbol "${oov}:1" \
            --add_symbol "${sos_eos}:-1"

            # Duplicated <sc> token may be counted for char token type,
            # so we shoud remove it
            if ${sot_avsr} && [ "${token_type}" = char ]; then
                cp ${token_list} ${token_list}".duplicated"
                awk '!seen[$0]++' ${token_list}".duplicated" > ${token_list}
                rm ${token_list}".duplicated"
            fi
    elif grep -q "whisper" <<< ${token_type}; then
        log "Stage 5: Generate whisper token_list from ${token_type} tokenizer"

        if ${sot_avsr}; then
            log "Error: not supported SOT training for whisper token_list"
            exit 2
        fi
        # The first symbol in token_list must be "<blank>" and the last must be also sos/eos:
        # 0 is reserved for CTC-blank for AVSR and also used as ignore-index in the other task
        echo ${token_list}
        ${python} -m espnet2.bin.whisper_export_vocabulary  \
            --whisper_model "${token_type}" \
            --output "${token_list}"
    elif [ "${token_type}" = hugging_face ]; then
        log "Stage 5: Generate hugging_face token_list from ${hugging_face_model_name_or_path}"

        if ${sot_avsr}; then
            log "Error: not supported SOT training for hugging_face token_list"
            exit 2
        fi
        # The first symbol in token_list must be "<blank>" and the last must be also sos/eos:
        # 0 is reserved for CTC-blank for AVSR and also used as ignore-index in the other task
        ${python} -m espnet2.bin.hugging_face_export_vocabulary  \
            --model_name_or_path "${hugging_face_model_name_or_path}" \
            --output "${token_list}"
    else
        log "Error: not supported --token_type '${token_type}'"
        exit 2
    fi

    # Create word-list for word-LM training
    if ${use_word_lm} && [ "${token_type}" != word ]; then
        log "Generate word level token_list from ${audio_data_feats}/lm_train.txt"
        ${python} -m espnet2.bin.tokenize_text \
            --token_type word \
            --input "${audio_data_feats}/lm_train.txt" --output "${lm_token_list}" \
            --field 2- \
            --cleaner "${cleaner}" \
            --g2p "${g2p}" \
            --write_vocabulary true \
            --vocabulary_size "${word_vocab_size}" \
            --add_symbol "${blank}:0" \
            --add_symbol "${oov}:1" \
            --add_symbol "${sos_eos}:-1"
    fi

fi


# ========================== Data preparation is done here. ==========================


if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ] && ! [[ " ${skip_stages} " =~ [[:space:]]6[[:space:]] ]]; then
    log "Stage 6: LM collect stats: train_set=${audio_data_feats}/lm_train.txt, dev_set=${lm_dev_text}"

    _opts=
    if [ -n "${lm_config}" ]; then
        # To generate the config file: e.g.
        #   % python3 -m espnet2.bin.lm_train --print_config --optim adam
        _opts+="--config ${lm_config} "
    fi

    # 1. Split the key file
    _logdir="${lm_stats_dir}/logdir"
    mkdir -p "${_logdir}"
    # Get the minimum number among ${nj} and the number lines of input files
    _nj=$(min "${nj}" "$(<${audio_data_feats}/lm_train.txt wc -l)" "$(<${lm_dev_text} wc -l)")

    key_file="${audio_data_feats}/lm_train.txt"
    split_scps=""
    for n in $(seq ${_nj}); do
        split_scps+=" ${_logdir}/train.${n}.scp"
    done
    # shellcheck disable=SC2086
    utils/split_scp.pl "${key_file}" ${split_scps}

    key_file="${lm_dev_text}"
    split_scps=""
    for n in $(seq ${_nj}); do
        split_scps+=" ${_logdir}/dev.${n}.scp"
    done
    # shellcheck disable=SC2086
    utils/split_scp.pl "${key_file}" ${split_scps}

    # 2. Generate run.sh
    log "Generate '${lm_stats_dir}/run.sh'. You can resume the process from stage 6 using this script"
    mkdir -p "${lm_stats_dir}"; echo "${run_args} --stage 6 \"\$@\"; exit \$?" > "${lm_stats_dir}/run.sh"; chmod +x "${lm_stats_dir}/run.sh"

    # 3. Submit jobs
    log "LM collect-stats started... log: '${_logdir}/stats.*.log'"
    # NOTE: --*_shape_file doesn't require length information if --batch_type=unsorted,
    #       but it's used only for deciding the sample ids.
    # shellcheck disable=SC2046,SC2086
    ${train_cmd} JOB=1:"${_nj}" "${_logdir}"/stats.JOB.log \
        ${python} -m espnet2.bin.lm_train \
            --collect_stats true \
            --use_preprocessor true \
            --bpemodel "${bpemodel}" \
            --token_type "${lm_token_type}"\
            --token_list "${lm_token_list}" \
            --non_linguistic_symbols "${nlsyms_txt}" \
            --cleaner "${cleaner}" \
            --g2p "${g2p}" \
            --train_data_path_and_name_and_type "${audio_data_feats}/lm_train.txt,text,text" \
            --valid_data_path_and_name_and_type "${lm_dev_text},text,text" \
            --train_shape_file "${_logdir}/train.JOB.scp" \
            --valid_shape_file "${_logdir}/dev.JOB.scp" \
            --output_dir "${_logdir}/stats.JOB" \
            ${_opts} ${lm_args} || { cat $(grep -l -i error "${_logdir}"/stats.*.log) ; exit 1; }

    # 4. Aggregate shape files
    _opts=
    for i in $(seq "${_nj}"); do
        _opts+="--input_dir ${_logdir}/stats.${i} "
    done
    # shellcheck disable=SC2086
    ${python} -m espnet2.bin.aggregate_stats_dirs ${_opts} --output_dir "${lm_stats_dir}"

    # Append the num-tokens at the last dimensions. This is used for batch-bins count
    <"${lm_stats_dir}/train/text_shape" \
        awk -v N="$(<${lm_token_list} wc -l)" '{ print $0 "," N }' \
        >"${lm_stats_dir}/train/text_shape.${lm_token_type}"

    <"${lm_stats_dir}/valid/text_shape" \
        awk -v N="$(<${lm_token_list} wc -l)" '{ print $0 "," N }' \
        >"${lm_stats_dir}/valid/text_shape.${lm_token_type}"
fi


if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ] && ! [[ " ${skip_stages} " =~ [[:space:]]7[[:space:]] ]]; then
    log "Stage 7: LM Training: train_set=${audio_data_feats}/lm_train.txt, dev_set=${lm_dev_text}"

    _opts=
    if [ -n "${lm_config}" ]; then
        # To generate the config file: e.g.
        #   % python3 -m espnet2.bin.lm_train --print_config --optim adam
        _opts+="--config ${lm_config} "
    fi

    if [ "${num_splits_lm}" -gt 1 ]; then
        # If you met a memory error when parsing text files, this option may help you.
        # The corpus is split into subsets and each subset is used for training one by one in order,
        # so the memory footprint can be limited to the memory required for each dataset.

        _split_dir="${lm_stats_dir}/splits${num_splits_lm}"
        if [ ! -f "${_split_dir}/.done" ]; then
            rm -f "${_split_dir}/.done"
            ${python} -m espnet2.bin.split_scps \
              --scps "${audio_data_feats}/lm_train.txt" "${lm_stats_dir}/train/text_shape.${lm_token_type}" \
              --num_splits "${num_splits_lm}" \
              --output_dir "${_split_dir}"
            touch "${_split_dir}/.done"
        else
            log "${_split_dir}/.done exists. Spliting is skipped"
        fi

        _opts+="--train_data_path_and_name_and_type ${_split_dir}/lm_train.txt,text,text "
        _opts+="--train_shape_file ${_split_dir}/text_shape.${lm_token_type} "
        _opts+="--multiple_iterator true "

    else
        _opts+="--train_data_path_and_name_and_type ${audio_data_feats}/lm_train.txt,text,text "
        _opts+="--train_shape_file ${lm_stats_dir}/train/text_shape.${lm_token_type} "
    fi

    # NOTE(kamo): --fold_length is used only if --batch_type=folded and it's ignored in the other case

    log "Generate '${lm_exp}/run.sh'. You can resume the process from stage 7 using this script"
    mkdir -p "${lm_exp}"; echo "${run_args} --stage 7 \"\$@\"; exit \$?" > "${lm_exp}/run.sh"; chmod +x "${lm_exp}/run.sh"

    log "LM training started... log: '${lm_exp}/train.log'"
    if echo "${cuda_cmd}" | grep -e queue.pl -e queue-freegpu.pl &> /dev/null; then
        # SGE can't include "/" in a job name
        jobname="$(basename ${lm_exp})"
    else
        jobname="${lm_exp}/train.log"
    fi

    # shellcheck disable=SC2086
    ${python} -m espnet2.bin.launch \
        --cmd "${cuda_cmd} --name ${jobname}" \
        --log "${lm_exp}"/train.log \
        --ngpu "${ngpu}" \
        --num_nodes "${num_nodes}" \
        --init_file_prefix "${lm_exp}"/.dist_init_ \
        --multiprocessing_distributed true -- \
        ${python} -m espnet2.bin.lm_train \
            --ngpu "${ngpu}" \
            --use_preprocessor true \
            --bpemodel "${bpemodel}" \
            --token_type "${lm_token_type}"\
            --token_list "${lm_token_list}" \
            --non_linguistic_symbols "${nlsyms_txt}" \
            --cleaner "${cleaner}" \
            --g2p "${g2p}" \
            --valid_data_path_and_name_and_type "${lm_dev_text},text,text" \
            --valid_shape_file "${lm_stats_dir}/valid/text_shape.${lm_token_type}" \
            --fold_length "${lm_fold_length}" \
            --resume true \
            --output_dir "${lm_exp}" \
            ${_opts} ${lm_args}

fi


if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ] && ! [[ " ${skip_stages} " =~ [[:space:]]8[[:space:]] ]]; then
    log "Stage 8: Calc perplexity: ${lm_test_text}"
    _opts=
    # TODO(kamo): Parallelize?
    log "Perplexity calculation started... log: '${lm_exp}/perplexity_test/lm_calc_perplexity.log'"
    # shellcheck disable=SC2086
    ${cuda_cmd} --gpu "${ngpu}" "${lm_exp}"/perplexity_test/lm_calc_perplexity.log \
        ${python} -m espnet2.bin.lm_calc_perplexity \
            --ngpu "${ngpu}" \
            --data_path_and_name_and_type "${lm_test_text},text,text" \
            --train_config "${lm_exp}"/config.yaml \
            --model_file "${lm_exp}/${inference_lm}" \
            --output_dir "${lm_exp}/perplexity_test" \
            ${_opts}
    log "PPL: ${lm_test_text}: $(cat ${lm_exp}/perplexity_test/ppl)"

fi


if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ] && ! [[ " ${skip_stages} " =~ [[:space:]]9[[:space:]] ]]; then
    log "Stage 9: Ngram Training: train_set=${audio_data_feats}/lm_train.txt"
    mkdir -p ${ngram_exp}
    cut -f 2- -d " " ${audio_data_feats}/lm_train.txt | lmplz -S "20%" --discount_fallback -o ${ngram_num} - >${ngram_exp}/${ngram_num}gram.arpa
    build_binary -s ${ngram_exp}/${ngram_num}gram.arpa ${ngram_exp}/${ngram_num}gram.bin
fi


# ========================== LM preparation is done here. ==========================


if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ] && ! [[ " ${skip_stages} " =~ [[:space:]]10[[:space:]] ]]; then
    _avsr_train_dir="${audio_data_feats}/${train_set}"
    _avsr_valid_dir="${audio_data_feats}/${valid_set}"
    log "Stage 10: AVSR collect stats: train_set=${_avsr_train_dir}, valid_set=${_avsr_valid_dir}"

    _opts=
    if [ -n "${avsr_config}" ]; then
        # To generate the config file: e.g.
        #   % python3 -m espnet2.bin.avsr_train --print_config --optim adam
        _opts+="--config ${avsr_config} "
    fi

    _audio_feats_type="$(<${_avsr_train_dir}/audio_feats_type)"
    _audio_format="$(cat ${_avsr_train_dir}/audio_format 2>/dev/null || echo ${audio_format})"
    if [ "${_audio_feats_type}" = raw ]; then
        _audio_scp=wav.scp
        if [[ "${_audio_format}" == *ark* ]]; then
            _audio_type=kaldi_ark
        else
            # "sound" supports "wav", "flac", etc.
            _audio_type=sound
        fi
        _opts+="--audio_frontend_conf fs=${fs} "
    else
        _audio_scp=feats.scp
        _audio_type=kaldi_ark
        _input_size="$(<${_avsr_train_dir}/feats_dim)"
        _opts+="--audio_input_size=${_input_size} "
    fi

    _visual_feats_type="$(<${_avsr_train_dir}/visual_feats_type)"
    _visual_format="$(cat ${_avsr_train_dir}/visual_format 2>/dev/null || echo ${visual_format})"
    if [ "${_visual_feats_type}" = raw ]; then
        _visual_scp=mp4.scp
        _visual_type=mp4
        # _opts+="--frontend_conf fps=${fps} "
    else
        echo "NotImplementedError: not `raw` visual feature is to be updated!"
        exit 1
    fi

    # 1. Split the key file
    _logdir="${avsr_stats_dir}/logdir"
    mkdir -p "${_logdir}"

    # Get the minimum number among ${nj} and the number lines of input files
    _nj=$(min "${nj}" "$(<${_avsr_train_dir}/${_audio_scp} wc -l)" "$(<${_avsr_valid_dir}/${_audio_scp} wc -l)")

    key_file="${_avsr_train_dir}/${_audio_scp}"
    split_scps=""
    for n in $(seq "${_nj}"); do
        split_scps+=" ${_logdir}/train.${n}.scp"
    done
    # shellcheck disable=SC2086
    utils/split_scp.pl "${key_file}" ${split_scps}

    key_file="${_avsr_valid_dir}/${_audio_scp}"
    split_scps=""
    for n in $(seq "${_nj}"); do
        split_scps+=" ${_logdir}/valid.${n}.scp"
    done
    # shellcheck disable=SC2086
    utils/split_scp.pl "${key_file}" ${split_scps}

    # 2. Generate run.sh
    log "Generate '${avsr_stats_dir}/run.sh'. You can resume the process from stage 10 using this script"
    mkdir -p "${avsr_stats_dir}"; echo "${run_args} --stage 10 \"\$@\"; exit \$?" > "${avsr_stats_dir}/run.sh"; chmod +x "${avsr_stats_dir}/run.sh"

    # 3. Submit jobs
    log "AVSR collect-stats started... log: '${_logdir}/stats.*.log'"

    # NOTE: --*_shape_file doesn't require length information if --batch_type=unsorted,
    #       but it's used only for deciding the sample ids.

    _opts+="--train_data_path_and_name_and_type ${_avsr_train_dir}/${_audio_scp},speech,${_audio_type} "
    _opts+="--valid_data_path_and_name_and_type ${_avsr_valid_dir}/${_audio_scp},speech,${_audio_type} "
    _opts+="--train_data_path_and_name_and_type ${_avsr_train_dir}/${_visual_scp},visual,${_visual_type} "
    _opts+="--valid_data_path_and_name_and_type ${_avsr_valid_dir}/${_visual_scp},visual,${_visual_type} "
    # shellcheck disable=SC2068
    for i in ${!ref_text_files[@]}; do
        _opts+="--train_data_path_and_name_and_type ${_avsr_train_dir}/${ref_text_files[$i]},${ref_text_names[$i]},text "
        _opts+="--valid_data_path_and_name_and_type ${_avsr_valid_dir}/${ref_text_files[$i]},${ref_text_names[$i]},text "
    done

    # shellcheck disable=SC2046,SC2086
    ${train_cmd} JOB=1:"${_nj}" "${_logdir}"/stats.JOB.log \
        ${python} -m espnet2.bin.${avsr_task}_train \
            --collect_stats true \
            --use_preprocessor true \
            --bpemodel "${bpemodel}" \
            --token_type "${token_type}" \
            --token_list "${token_list}" \
            --non_linguistic_symbols "${nlsyms_txt}" \
            --cleaner "${cleaner}" \
            --g2p "${g2p}" \
            --train_shape_file "${_logdir}/train.JOB.scp" \
            --valid_shape_file "${_logdir}/valid.JOB.scp" \
            --output_dir "${_logdir}/stats.JOB" \
            ${_opts} ${avsr_args} || { cat $(grep -l -i error "${_logdir}"/stats.*.log) ; exit 1; }

    # 4. Aggregate shape files
    _opts=
    for i in $(seq "${_nj}"); do
        _opts+="--input_dir ${_logdir}/stats.${i} "
    done
    if [ "${feats_normalize}" != global_mvn ]; then
        # Skip summerizaing stats if not using global MVN
        _opts+="--skip_sum_stats"
    fi
    # shellcheck disable=SC2086
    ${python} -m espnet2.bin.aggregate_stats_dirs ${_opts} --output_dir "${avsr_stats_dir}"

    # Append the num-tokens at the last dimensions. This is used for batch-bins count
    # shellcheck disable=SC2068
    for ref_txt in ${ref_text_names[@]}; do
        <"${avsr_stats_dir}/train/${ref_txt}_shape" \
            awk -v N="$(<${token_list} wc -l)" '{ print $0 "," N }' \
            >"${avsr_stats_dir}/train/${ref_txt}_shape.${token_type}"

        <"${avsr_stats_dir}/valid/${ref_txt}_shape" \
            awk -v N="$(<${token_list} wc -l)" '{ print $0 "," N }' \
            >"${avsr_stats_dir}/valid/${ref_txt}_shape.${token_type}"
    done
fi


if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ] && ! [[ " ${skip_stages} " =~ [[:space:]]11[[:space:]] ]]; then
    _avsr_train_dir="${audio_data_feats}/${train_set}"
    _avsr_valid_dir="${audio_data_feats}/${valid_set}"
    log "Stage 11: AVSR Training: train_set=${_avsr_train_dir}, valid_set=${_avsr_valid_dir}"

    _opts=
    if [ -n "${avsr_config}" ]; then
        # To generate the config file: e.g.
        #   % python3 -m espnet2.bin.avsr_train --print_config --optim adam
        _opts+="--config ${avsr_config} "
    fi

    _audio_feats_type="$(<${_avsr_train_dir}/audio_feats_type)"
    _audio_format="$(cat ${_avsr_train_dir}/audio_format 2>/dev/null || echo ${audio_format})"
    if [ "${_audio_feats_type}" = raw ]; then
        _audio_scp=wav.scp
        # "sound" supports "wav", "flac", etc.
        if [[ "${_audio_format}" == *ark* ]]; then
            _audio_type=kaldi_ark
        elif [[ "${_audio_format}" == *multi* ]]; then
            _audio_type=multi_columns_sound
        else
            _audio_type=sound
        fi
        _fold_length="$((avsr_speech_fold_length * 100))"
        _opts+="--audio_frontend_conf fs=${fs} "
    else
        _audio_scp=feats.scp
        _audio_type=kaldi_ark
        _fold_length="${avsr_speech_fold_length}"
        _input_size="$(<${_avsr_train_dir}/feats_dim)"
        _opts+="--input_size=${_input_size} "

    fi
    
    _visual_feats_type="$(<${_avsr_train_dir}/visual_feats_type)"
    _visual_format="$(cat ${_avsr_train_dir}/visual_format 2>/dev/null || echo ${visual_format})"
    if [ "${_visual_feats_type}" = raw ]; then
        _visual_scp=mp4.scp
        _visual_type=mp4
        # _opts+="--frontend_conf fps=${fps} "
    else
        echo "NotImplementedError: not `raw` visual feature is to be updated!"
        exit 1
    fi

    if [ "${feats_normalize}" = global_mvn ]; then
        # Default normalization is utterance_mvn and changes to global_mvn
        _opts+="--audio_normalize=global_mvn --audio_normalize_conf stats_file=${avsr_stats_dir}/train/audio_feats_stats.npz "
        _opts+="--visual_normalize=global_mvn --visual_normalize_conf stats_file=${avsr_stats_dir}/train/visual_feats_stats.npz "
    fi

    if [ "${num_splits_avsr}" -gt 1 ]; then
        # If you met a memory error when parsing text files, this option may help you.
        # The corpus is split into subsets and each subset is used for training one by one in order,
        # so the memory footprint can be limited to the memory required for each dataset.

        _split_dir="${avsr_stats_dir}/splits${num_splits_avsr}"
        if [ ! -f "${_split_dir}/.done" ]; then
            rm -f "${_split_dir}/.done"
            ${python} -m espnet2.bin.split_scps \
              --scps \
                  "${_avsr_train_dir}/${_audio_scp}" \
                  "${_avsr_train_dir}/${_visual_scp}" \
                  "${_avsr_train_dir}/text" \
                  "${avsr_stats_dir}/train/speech_shape" \
                  "${avsr_stats_dir}/train/visual_shape" \
                  "${avsr_stats_dir}/train/text_shape.${token_type}" \
              --num_splits "${num_splits_avsr}" \
              --output_dir "${_split_dir}"
            touch "${_split_dir}/.done"
        else
            log "${_split_dir}/.done exists. Spliting is skipped"
        fi

        _opts+="--train_data_path_and_name_and_type ${_split_dir}/${_audio_scp},speech,${_speech_type} "
        _opts+="--train_data_path_and_name_and_type ${_split_dir}/${_visual_scp},visual,${_visual_type} "
        _opts+="--train_shape_file ${_split_dir}/speech_shape "
        _opts+="--train_shape_file ${_split_dir}/visual_shape "
        # shellcheck disable=SC2068
        for i in ${!ref_text_names[@]}; do
            _opts+="--fold_length ${avsr_text_fold_length} "
            _opts+="--train_data_path_and_name_and_type ${_split_dir}/${ref_text_files[$i]},${ref_text_names[$i]},text "
            _opts+="--train_shape_file ${_split_dir}/${ref_text_names[$i]}_shape.${token_type} "
        done
        _opts+="--multiple_iterator true "

    else
        _opts+="--train_data_path_and_name_and_type ${_avsr_train_dir}/${_audio_scp},speech,${_audio_type} "
        _opts+="--train_data_path_and_name_and_type ${_avsr_train_dir}/${_visual_scp},visual,${_visual_type} "
        _opts+="--train_shape_file ${avsr_stats_dir}/train/speech_shape "
        _opts+="--train_shape_file ${avsr_stats_dir}/train/visual_shape "
        
        read -r -a aux_list <<< "$auxiliary_data_tags"
        if [ ${#aux_list[@]} != 0 ]; then
            _opts+="--allow_variable_data_keys True "
            for aux_dset in "${aux_list[@]}"; do
                 _opts+="--train_data_path_and_name_and_type ${_avsr_train_dir}/${aux_dset},text,text "
            done
        fi
        # shellcheck disable=SC2068
        for i in ${!ref_text_names[@]}; do
            _opts+="--fold_length ${avsr_text_fold_length} "
            _opts+="--train_data_path_and_name_and_type ${_avsr_train_dir}/${ref_text_files[$i]},${ref_text_names[$i]},text "
            _opts+="--train_shape_file ${avsr_stats_dir}/train/${ref_text_names[$i]}_shape.${token_type} "
        done
    fi

    # shellcheck disable=SC2068
    for i in ${!ref_text_names[@]}; do
        _opts+="--valid_data_path_and_name_and_type ${_avsr_valid_dir}/${ref_text_files[$i]},${ref_text_names[$i]},text "
        _opts+="--valid_shape_file ${avsr_stats_dir}/valid/${ref_text_names[$i]}_shape.${token_type} "
    done

    log "Generate '${avsr_exp}/run.sh'. You can resume the process from stage 11 using this script"
    mkdir -p "${avsr_exp}"; echo "${run_args} --stage 11 \"\$@\"; exit \$?" > "${avsr_exp}/run.sh"; chmod +x "${avsr_exp}/run.sh"

    # NOTE(kamo): --fold_length is used only if --batch_type=folded and it's ignored in the other case
    log "AVSR training started... log: '${avsr_exp}/train.log'"
    if echo "${cuda_cmd}" | grep -e queue.pl -e queue-freegpu.pl &> /dev/null; then
        # SGE can't include "/" in a job name
        jobname="$(basename ${avsr_exp})"
    else
        jobname="${avsr_exp}/train.log"
    fi

    # shellcheck disable=SC2086
    ${python} -m espnet2.bin.launch \
        --cmd "${cuda_cmd} --name ${jobname}" \
        --log "${avsr_exp}"/train.log \
        --ngpu "${ngpu}" \
        --num_nodes "${num_nodes}" \
        --init_file_prefix "${avsr_exp}"/.dist_init_ \
        --multiprocessing_distributed true -- \
        ${python} -m espnet2.bin.${avsr_task}_train \
            --use_preprocessor true \
            --bpemodel "${bpemodel}" \
            --token_type "${token_type}" \
            --token_list "${token_list}" \
            --non_linguistic_symbols "${nlsyms_txt}" \
            --cleaner "${cleaner}" \
            --g2p "${g2p}" \
            --valid_data_path_and_name_and_type "${_avsr_valid_dir}/${_audio_scp},speech,${_audio_type}" \
            --valid_shape_file "${avsr_stats_dir}/valid/speech_shape" \
            --valid_data_path_and_name_and_type "${_avsr_valid_dir}/${_visual_scp},visual,${_visual_type}" \
            --valid_shape_file "${avsr_stats_dir}/valid/visual_shape" \
            --resume true \
            ${pretrained_model:+--init_param $pretrained_model} \
            --ignore_init_mismatch ${ignore_init_mismatch} \
            --fold_length "${_fold_length}" \
            --output_dir "${avsr_exp}" \
            ${_opts} ${avsr_args}

fi


if [ -n "${download_model}" ]; then
    log "Use ${download_model} for decoding and evaluation"
    avsr_exp="${expdir}/${download_model}"
    mkdir -p "${avsr_exp}"

    # If the model already exists, you can skip downloading
    espnet_model_zoo_download --unpack true "${download_model}" > "${avsr_exp}/config.txt"

    # Get the path of each file
    _avsr_model_file=$(<"${avsr_exp}/config.txt" sed -e "s/.*'avsr_model_file': '\([^']*\)'.*$/\1/")
    _avsr_train_config=$(<"${avsr_exp}/config.txt" sed -e "s/.*'avsr_train_config': '\([^']*\)'.*$/\1/")

    # Create symbolic links
    ln -sf "${_avsr_model_file}" "${avsr_exp}"
    ln -sf "${_avsr_train_config}" "${avsr_exp}"
    inference_avsr_model=$(basename "${_avsr_model_file}")

    if [ "$(<${avsr_exp}/config.txt grep -c lm_file)" -gt 0 ]; then
        _lm_file=$(<"${avsr_exp}/config.txt" sed -e "s/.*'lm_file': '\([^']*\)'.*$/\1/")
        _lm_train_config=$(<"${avsr_exp}/config.txt" sed -e "s/.*'lm_train_config': '\([^']*\)'.*$/\1/")

        lm_exp="${expdir}/${download_model}/lm"
        mkdir -p "${lm_exp}"

        ln -sf "${_lm_file}" "${lm_exp}"
        ln -sf "${_lm_train_config}" "${lm_exp}"
        inference_lm=$(basename "${_lm_file}")
    fi

fi


if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ] && ! [[ " ${skip_stages} " =~ [[:space:]]12[[:space:]] ]]; then
    log "Stage 12: Decoding: training_dir=${avsr_exp}"

    if ${gpu_inference}; then
        _cmd="${cuda_cmd}"
        _ngpu=1
    else
        _cmd="${decode_cmd}"
        _ngpu=0
    fi

    _opts=
    if [ -n "${inference_config}" ]; then
        _opts+="--config ${inference_config} "
    fi
    if "${use_lm}"; then
        if "${use_word_lm}"; then
            _opts+="--word_lm_train_config ${lm_exp}/config.yaml "
            _opts+="--word_lm_file ${lm_exp}/${inference_lm} "
        else
            _opts+="--lm_train_config ${lm_exp}/config.yaml "
            _opts+="--lm_file ${lm_exp}/${inference_lm} "
        fi
    fi
    if "${use_ngram}"; then
         _opts+="--ngram_file ${ngram_exp}/${inference_ngram}"
    fi

    # 2. Generate run.sh
    log "Generate '${avsr_exp}/${inference_tag}/run.sh'. You can resume the process from stage 12 using this script"
    mkdir -p "${avsr_exp}/${inference_tag}"; echo "${run_args} --stage 12 \"\$@\"; exit \$?" > "${avsr_exp}/${inference_tag}/run.sh"; chmod +x "${avsr_exp}/${inference_tag}/run.sh"

    inference_bin_tag=""
    if [ ${avsr_task} == "avsr" ]; then
        if "${use_k2}"; then
            # Now only _nj=1 is verified if using k2
            inference_bin_tag="_k2"

            _opts+="--is_ctc_decoding ${k2_ctc_decoding} "
            _opts+="--use_nbest_rescoring ${use_nbest_rescoring} "
            _opts+="--num_paths ${num_paths} "
            _opts+="--nll_batch_size ${nll_batch_size} "
            _opts+="--k2_config ${k2_config} "
        elif "${use_streaming}"; then
            inference_bin_tag="_streaming"
        elif "${use_maskctc}"; then
            inference_bin_tag="_maskctc"
        fi
    fi

    if "${eval_valid_set}"; then
        _dsets="org/${valid_set} ${test_sets}"
    else
        _dsets="${test_sets}"
    fi
    for dset in ${_dsets}; do
        _data="${audio_data_feats}/${dset}"
        _dir="${avsr_exp}/${inference_tag}/${dset}"
        _logdir="${_dir}/logdir"
        mkdir -p "${_logdir}"

        _audio_feats_type="$(<${_data}/audio_feats_type)"
        _audio_format="$(cat ${_data}/audio_format 2>/dev/null || echo ${audio_format})"
        if [ "${_audio_feats_type}" = raw ]; then
            _audio_scp=wav.scp
            if [[ "${audio_format}" == *ark* ]]; then
                _audio_type=kaldi_ark
            elif [[ "${_audio_format}" == *multi* ]]; then
                _audio_type=multi_columns_sound
            else
                _audio_type=sound
            fi
        else
            _audio_scp=feats.scp
            _audio_type=kaldi_ark
        fi

        _visual_feats_type="$(<${_data}/visual_feats_type)"
        _visual_format="$(cat ${_data}/visual_format 2>/dev/null || echo ${visual_format})"
        if [ "${_visual_feats_type}" = raw ]; then
            _visual_scp=mp4.scp
            _visual_type=mp4
            # _opts+="--frontend_conf fps=${fps} "
        else
            echo "NotImplementedError: not `raw` visual feature is to be updated!"
            exit 1
        fi

        # 1. Split the key file
        key_file=${_data}/${_audio_scp}
        split_scps=""
        if "${use_k2}"; then
          # Now only _nj=1 is verified if using k2
          _nj=1
        else
          _nj=$(min "${inference_nj}" "$(<${key_file} wc -l)")
        fi

        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/keys.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        # 2. Submit decoding jobs
        log "Decoding started... log: '${_logdir}/avsr_inference.*.log'"
        rm -f "${_logdir}/*.log"
        # shellcheck disable=SC2046,SC2086
        ${_cmd} --gpu "${_ngpu}" JOB=1:"${_nj}" "${_logdir}"/avsr_inference.JOB.log \
            ${python} -m espnet2.bin.${avsr_task}_inference${inference_bin_tag} \
                --batch_size ${batch_size} \
                --ngpu "${_ngpu}" \
                --data_path_and_name_and_type "${_data}/${_audio_scp},speech,${_audio_type}" \
                --data_path_and_name_and_type "${_data}/${_visual_scp},visual,${_visual_type}" \
                --key_file "${_logdir}"/keys.JOB.scp \
                --avsr_train_config "${avsr_exp}"/config.yaml \
                --avsr_model_file "${avsr_exp}"/"${inference_avsr_model}" \
                --output_dir "${_logdir}"/output.JOB \
                ${_opts} ${inference_args} || { cat $(grep -l -i error "${_logdir}"/avsr_inference.*.log) ; exit 1; }

        # 3. Calculate and report RTF based on decoding logs
        if [ ${avsr_task} == "avsr" ] && [ -z ${inference_bin_tag} ]; then
            log "Calculating RTF & latency... log: '${_logdir}/calculate_rtf.log'"
            rm -f "${_logdir}"/calculate_rtf.log
            _fs=$(python3 -c "import humanfriendly as h;print(h.parse_size('${fs}'))")
            _sample_shift=$(python3 -c "print(1 / ${_fs} * 1000)") # in ms
            ${_cmd} JOB=1 "${_logdir}"/calculate_rtf.log \
                pyscripts/utils/calculate_rtf.py \
                    --log-dir ${_logdir} \
                    --log-name "avsr_inference" \
                    --input-shift ${_sample_shift} \
                    --start-times-marker "speech length" \
                    --end-times-marker "best hypo" \
                    --inf-num ${num_inf} || { cat "${_logdir}"/calculate_rtf.log; exit 1; }
        fi

        # 4. Concatenates the output files from each jobs
        # shellcheck disable=SC2068
        for ref_txt in ${ref_text_files[@]}; do
            suffix=$(echo ${ref_txt} | sed 's/text//')
            for f in token token_int score text; do
                if [ -f "${_logdir}/output.1/1best_recog/${f}${suffix}" ]; then
                    for i in $(seq "${_nj}"); do
                        cat "${_logdir}/output.${i}/1best_recog/${f}${suffix}"
                    done | sort -k1 >"${_dir}/${f}${suffix}"
                fi
            done
        done

    done
fi


if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ] && ! [[ " ${skip_stages} " =~ [[:space:]]13[[:space:]] ]]; then
    log "Stage 13: Scoring"
    if [ "${token_type}" = phn ]; then
        log "Error: Not implemented for token_type=phn"
        exit 1
    fi

    if "${eval_valid_set}"; then
        _dsets="org/${valid_set} ${test_sets}"
    else
        _dsets="${test_sets}"
    fi
    for dset in ${_dsets}; do
        _data="${audio_data_feats}/${dset}"
        _dir="${avsr_exp}/${inference_tag}/${dset}"

        for _tok_type in "char" "word" "bpe"; do
            [ "${_tok_type}" = bpe ] && [ ! -f "${bpemodel}" ] && continue

            _opts="--token_type ${_tok_type} "
            if [ "${_tok_type}" = "char" ] || [ "${_tok_type}" = "word" ]; then
                _type="${_tok_type:0:1}er"
                _opts+="--non_linguistic_symbols ${nlsyms_txt} "
                _opts+="--remove_non_linguistic_symbols true "

            elif [ "${_tok_type}" = "bpe" ]; then
                _type="ter"
                _opts+="--bpemodel ${bpemodel} "

            else
                log "Error: unsupported token type ${_tok_type}"
            fi

            _scoredir="${_dir}/score_${_type}"
            mkdir -p "${_scoredir}"

            # shellcheck disable=SC2068
            for ref_txt in ${ref_text_files[@]}; do
                # Note(simpleoier): to get the suffix after text, e.g. "text_spk1" -> "_spk1"
                suffix=$(echo ${ref_txt} | sed 's/text//')

                # Tokenize text to ${_tok_type} level
                paste \
                    <(<"${_data}/${ref_txt}" \
                        ${python} -m espnet2.bin.tokenize_text  \
                            -f 2- --input - --output - \
                            --cleaner "${cleaner}" \
                            ${_opts} \
                            ) \
                    <(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
                        >"${_scoredir}/ref${suffix:-${suffix}}.trn"

                # NOTE(kamo): Don't use cleaner for hyp
                paste \
                    <(<"${_dir}/${ref_txt}"  \
                        ${python} -m espnet2.bin.tokenize_text  \
                            -f 2- --input - --output - \
                            ${_opts} \
                            --cleaner "${hyp_cleaner}" \
                            ) \
                    <(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
                        >"${_scoredir}/hyp${suffix:-${suffix}}.trn"

            done

            # Note(simpleoier): score across all possible permutations
            if [ ${num_ref} -gt 1 ] && [ -n "${suffix}" ]; then
                for i in $(seq ${num_ref}); do
                    for j in $(seq ${num_inf}); do
                        sclite \
                            ${score_opts} \
                            -r "${_scoredir}/ref_spk${i}.trn" trn \
                            -h "${_scoredir}/hyp_spk${j}.trn" trn \
                            -i rm -o all stdout > "${_scoredir}/result_r${i}h${j}.txt"
                    done
                done
                # Generate the oracle permutation hyp.trn and ref.trn
                scripts/utils/eval_perm_free_error.py --num-spkrs ${num_ref} \
                    --results-dir ${_scoredir}
            fi

            sclite \
                ${score_opts} \
                -r "${_scoredir}/ref.trn" trn \
                -h "${_scoredir}/hyp.trn" trn \
                -i rm -o all stdout > "${_scoredir}/result.txt"

            log "Write ${_type} result in ${_scoredir}/result.txt"
            grep -e Avg -e SPKR -m 2 "${_scoredir}/result.txt"
        done
    done

    [ -f local/score.sh ] && local/score.sh ${local_score_opts} "${avsr_exp}"

    # Show results in Markdown syntax
    scripts/utils/show_avsr_result.sh "${avsr_exp}" > "${avsr_exp}"/RESULTS.md
    cat "${avsr_exp}"/RESULTS.md

fi


packed_model="${avsr_exp}/${avsr_exp##*/}_${inference_avsr_model%.*}.zip"
if [ ${stage} -le 14 ] && [ ${stop_stage} -ge 14 ] && ! [[ " ${skip_stages} " =~ [[:space:]]14[[:space:]] ]]; then
    log "Stage 14: Pack model: ${packed_model}"

    _opts=
    if "${use_lm}"; then
        _opts+="--lm_train_config ${lm_exp}/config.yaml "
        _opts+="--lm_file ${lm_exp}/${inference_lm} "
        _opts+="--option ${lm_exp}/perplexity_test/ppl "
        _opts+="--option ${lm_exp}/images "
    fi
    if [ "${feats_normalize}" = global_mvn ]; then
        _opts+="--option ${avsr_stats_dir}/train/feats_stats.npz "
    fi
    if [ "${token_type}" = bpe ]; then
        _opts+="--option ${bpemodel} "
    fi
    if [ "${nlsyms_txt}" != none ]; then
        _opts+="--option ${nlsyms_txt} "
    fi
    # shellcheck disable=SC2086
    ${python} -m espnet2.bin.pack avsr \
        --avsr_train_config "${avsr_exp}"/config.yaml \
        --avsr_model_file "${avsr_exp}"/"${inference_avsr_model}" \
        ${_opts} \
        --option "${avsr_exp}"/RESULTS.md \
        --option "${avsr_exp}"/images \
        --outpath "${packed_model}"
fi


if [ ${stage} -le 15 ] && [ ${stop_stage} -ge 15 ] && ! [[ " ${skip_stages} " =~ [[:space:]]15[[:space:]] ]]; then
    log "Stage 15: Upload model to Zenodo: ${packed_model}"
    log "Warning: Upload model to Zenodo will be deprecated. We encourage to use Hugging Face"

    # To upload your model, you need to do:
    #   1. Sign up to Zenodo: https://zenodo.org/
    #   2. Create access token: https://zenodo.org/account/settings/applications/tokens/new/
    #   3. Set your environment: % export ACCESS_TOKEN="<your token>"

    if command -v git &> /dev/null; then
        _creator_name="$(git config user.name)"
        _checkout="
git checkout $(git show -s --format=%H)"

    else
        _creator_name="$(whoami)"
        _checkout=""
    fi
    # /some/where/espnet/egs2/foo/avsr1/ -> foo/avsr1
    _task="$(pwd | rev | cut -d/ -f2 | rev)"
    # foo/avsr1 -> foo
    _corpus="${_task%/*}"
    _model_name="${_creator_name}/${_corpus}_$(basename ${packed_model} .zip)"

    # Generate description file
    cat << EOF > "${avsr_exp}"/description
This model was trained by ${_creator_name} using ${_task} recipe in <a href="https://github.com/espnet/espnet/">espnet</a>.
<p>&nbsp;</p>
<ul>
<li><strong>Python API</strong><pre><code class="language-python">See https://github.com/espnet/espnet_model_zoo</code></pre></li>
<li><strong>Evaluate in the recipe</strong><pre>
<code class="language-bash">git clone https://github.com/espnet/espnet
cd espnet${_checkout}
pip install -e .
cd $(pwd | rev | cut -d/ -f1-3 | rev)
./run.sh --skip_data_prep false --skip_train true --download_model ${_model_name}</code>
</pre></li>
<li><strong>Results</strong><pre><code>$(cat "${avsr_exp}"/RESULTS.md)</code></pre></li>
<li><strong>AVSR config</strong><pre><code>$(cat "${avsr_exp}"/config.yaml)</code></pre></li>
<li><strong>LM config</strong><pre><code>$(if ${use_lm}; then cat "${lm_exp}"/config.yaml; else echo NONE; fi)</code></pre></li>
</ul>
EOF

    # NOTE(kamo): The model file is uploaded here, but not published yet.
    #   Please confirm your record at Zenodo and publish it by yourself.

    # shellcheck disable=SC2086
    espnet_model_zoo_upload \
        --file "${packed_model}" \
        --title "ESPnet2 pretrained model, ${_model_name}, fs=${fs}, lang=${lang}" \
        --description_file "${avsr_exp}"/description \
        --creator_name "${_creator_name}" \
        --license "CC-BY-4.0" \
        --use_sandbox false \
        --publish false
fi


if [ ${stage} -le 16 ] && [ ${stop_stage} -ge 16 ] && ! [[ " ${skip_stages} " =~ [[:space:]]16[[:space:]] ]]; then
    [ -z "${hf_repo}" ] && \
        log "ERROR: You need to setup the variable hf_repo with the name of the repository located at HuggingFace, follow the following steps described here https://github.com/espnet/espnet/blob/master/CONTRIBUTING.md#132-espnet2-recipes" && \
    exit 1
    log "Stage 16: Upload model to HuggingFace: ${hf_repo}"

    gitlfs=$(git lfs --version 2> /dev/null || true)
    [ -z "${gitlfs}" ] && \
        log "ERROR: You need to install git-lfs first" && \
        exit 1

    dir_repo=${expdir}/hf_${hf_repo//"/"/"_"}
    [ ! -d "${dir_repo}" ] && git clone https://huggingface.co/${hf_repo} ${dir_repo}

    if command -v git &> /dev/null; then
        _creator_name="$(git config user.name)"
        _checkout="git checkout $(git show -s --format=%H)"
    else
        _creator_name="$(whoami)"
        _checkout=""
    fi
    # /some/where/espnet/egs2/foo/avsr1/ -> foo/avsr1
    _task="$(pwd | rev | cut -d/ -f2 | rev)"
    # foo/avsr1 -> foo
    _corpus="${_task%/*}"
    _model_name="${_creator_name}/${_corpus}_$(basename ${packed_model} .zip)"

    # copy files in ${dir_repo}
    unzip -o ${packed_model} -d ${dir_repo}
    # Generate description file
    # shellcheck disable=SC2034
    hf_task=automatic-speech-recognition
    # shellcheck disable=SC2034
    espnet_task=AVSR
    # shellcheck disable=SC2034
    task_exp=${avsr_exp}
    eval "echo \"$(cat scripts/utils/TEMPLATE_HF_Readme.md)\"" > "${dir_repo}"/README.md

    this_folder=${PWD}
    cd ${dir_repo}
    if [ -n "$(git status --porcelain)" ]; then
        git add .
        git commit -m "Update model"
    fi
    git push
    cd ${this_folder}
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
