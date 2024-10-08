#!/usr/bin/env bash

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

set -eou pipefail

nj=15
stage=0
stop_stage=100

# Path to your local myvoice_train.jsonl.gz file
local_data_path="/content/drive/MyDrive/ColabData/KWS/kws_create_dataset/myvoice_train.jsonl.gz"

dl_dir=$PWD/download
lang_char_dir=data/lang_char

. shared/parse_options.sh || exit 1

# All files generated by this script are saved in "data".
# You can safely remove "data" and rerun this script to regenerate it.
mkdir -p data

log() {
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

log "dl_dir: $dl_dir"

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "Stage 0: Copy local data and download musan if needed"
  mkdir -p data/manifests
  cp $local_data_path data/manifests/cuts_train.jsonl.gz
  cp /content/drive/MyDrive/ColabData/KWS/kws_create_dataset/*.wav /content/icefall/egs/wenetspeech/ASR/

  if [ ! -d $dl_dir/musan ]; then
    log "Downloading musan dataset"
    lhotse download musan $dl_dir
  fi
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Prepare musan manifest"
  mkdir -p data/manifests
  lhotse prepare musan $dl_dir/musan data/manifests
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: Preprocess local manifest"
  if [ ! -f data/fbank/.preprocess_complete ]; then
    python3 ./local/preprocess_wenetspeech.py --perturb-speed True
    touch data/fbank/.preprocess_complete
  fi
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "Stage 3: Combine features"
  if [ ! -f data/fbank/cuts_train.jsonl.gz ]; then
    cp data/manifests/cuts_train.jsonl.gz data/fbank/cuts_train.jsonl.gz
  fi
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  log "Stage 4: Prepare char based lang"
  mkdir -p $lang_char_dir

  if ! which jq; then
      echo "This script is intended to be used with jq but you have not installed jq
      Note: in Linux, you can install jq with the following command:
      1. wget -O jq https://github.com/stedolan/jq/releases/download/jq-1.6/jq-linux64
      2. chmod +x ./jq
      3. cp jq /usr/bin" && exit 1
  fi
  if [ ! -f $lang_char_dir/text ] || [ ! -s $lang_char_dir/text ]; then
    log "Prepare text."
    gunzip -c data/manifests/cuts_train.jsonl.gz \
      | jq '.text' | sed 's/"//g' \
      | ./local/text2token.py -t "char" > $lang_char_dir/text
  fi

  # The implementation of chinese word segmentation for text,
  # and it will take about 15 minutes.
  if [ ! -f $lang_char_dir/text_words_segmentation ]; then
    python3 ./local/text2segments.py \
      --num-process $nj \
      --input-file $lang_char_dir/text \
      --output-file $lang_char_dir/text_words_segmentation
  fi

  cat $lang_char_dir/text_words_segmentation | sed 's/ /\n/g' \
    | sort -u | sed '/^$/d' | uniq > $lang_char_dir/words_no_ids.txt

  if [ ! -f $lang_char_dir/words.txt ]; then
    python3 ./local/prepare_words.py \
      --input-file $lang_char_dir/words_no_ids.txt \
      --output-file $lang_char_dir/words.txt
  fi
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  log "Stage 5: Prepare char based L_disambig.pt"
  if [ ! -f data/lang_char/L_disambig.pt ]; then
    python3 ./local/prepare_char.py \
      --lang-dir data/lang_char
  fi
fi

# Add any additional stages you need for your specific use case

log "Data preparation completed."
