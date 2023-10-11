#!/usr/bin/env bash
set -euo pipefail
SECONDS=0
log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
help_message=$(cat << EOF
Usage: $0 <in-mp4.scp> <out-datadir> [<logdir> [<outdir>]]
e.g.
$0 data/test/mp4.scp data/test_format/

Format 'mp4.scp': In short words,
changing "kaldi-datadir" to "modified-kaldi-datadir"

The 'mp4.scp' format in kaldi is very flexible,
e.g. It can use unix-pipe as describing that mp4 file,
but it sometime looks confusing and make scripts more complex.
This tools creates actual mp4 files from 'mp4.scp'
and also segments mp4 files using 'segments'.

Options
  --fps <fps>
  --segments <segments>
  --nj <nj>
  --cmd <cmd>
EOF
)

out_filename=mp4.scp
cmd=utils/run.pl
nj=30
fps=none
segments=

ref_channels=
utt2ref_channels=

visual_format=mp4
write_utt2num_samples=true
vad_based_trim=
multi_columns_input=false
multi_columns_output=false

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 2 ] && [ $# -ne 3 ] && [ $# -ne 4 ]; then
    log "${help_message}"
    log "Error: invalid command line arguments"
    exit 1
fi

. ./path.sh  # Setup the environment

scp=$1
if [ ! -f "${scp}" ]; then
    log "${help_message}"
    echo "$0: Error: No such file: ${scp}"
    exit 1
fi
dir=$2


if [ $# -eq 2 ]; then
    logdir=${dir}/logs
    outdir=${dir}/data

elif [ $# -eq 3 ]; then
    logdir=$3
    outdir=${dir}/data

elif [ $# -eq 4 ]; then
    logdir=$3
    outdir=$4
fi


mkdir -p ${logdir}

rm -f "${dir}/${out_filename}"


opts=
if [ -n "${utt2ref_channels}" ]; then
    opts="--utt2ref-channels ${utt2ref_channels} "
elif [ -n "${ref_channels}" ]; then
    opts="--ref-channels ${ref_channels} "
fi

if [ -n "${vad_based_trim}" ]; then
    opts="--vad_based_trim ${vad_based_trim} "
fi

if [ -n "${segments}" ]; then
    log "[info]: using ${segments}"
    nutt=$(<${segments} wc -l)
    nj=$((nj<nutt?nj:nutt))

    split_segments=""
    for n in $(seq ${nj}); do
        split_segments="${split_segments} ${logdir}/segments.${n}"
    done
    utils/split_scp.pl "${segments}" ${split_segments}

    # shellcheck disable=SC2046
    ${cmd} "JOB=1:${nj}" "${logdir}/format_mp4_scp.JOB.log" \
        pyscripts/audio/format_mp4_scp.py \
            ${opts} \
            --fps ${fps} \
            --visual-format "${visual_format}" \
            "--segment=${logdir}/segments.JOB" \
            --multi-columns-input "${multi_columns_input}" \
            --multi-columns-output "${multi_columns_output}" \
            "${scp}" "${outdir}/format.JOB" || { cat $(grep -l -i error "${logdir}"/format_mp4_scp.*.log) ; exit 1; }
else
    log "[info]: without segments"
    nutt=$(<${scp} wc -l)
    nj=$((nj<nutt?nj:nutt))

    split_scps=""
    for n in $(seq ${nj}); do
        split_scps="${split_scps} ${logdir}/mp4.${n}.scp"
    done
    
    utils/split_scp.pl "${scp}" ${split_scps}
    # shellcheck disable=SC2046
    ${cmd} "JOB=1:${nj}" "${logdir}/format_mp4_scp.JOB.log" \
        pyscripts/visual/format_mp4_scp.py \
        ${opts} \
        --fps "${fps}" \
        --visual-format "${visual_format}" \
        --multi-columns-input "${multi_columns_input}" \
        --multi-columns-output "${multi_columns_output}" \
        "${logdir}/mp4.JOB.scp" "${outdir}/format.JOB" || { cat $(grep -l -i error "${logdir}"/format_mp4_scp.*.log) ; exit 1; }
fi

# Workaround for the NFS problem
ls ${outdir}/format.* > /dev/null

# concatenate the .scp files together.
for n in $(seq ${nj}); do
    cat "${outdir}/format.${n}/mp4.scp" || exit 1;
done > "${dir}/${out_filename}" || exit 1

if "${write_utt2num_samples}"; then
    for n in $(seq ${nj}); do
        cat "${outdir}/format.${n}/utt2num_samples" || exit 1;
    done > "${dir}/utt2num_samples"  || exit 1
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
