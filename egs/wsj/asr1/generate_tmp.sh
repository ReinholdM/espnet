#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=0        # start from 0 if you need to start from data preparation
stop_stage=100
ngpu=4         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1

N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot
seed=1

# feature configuration
do_delta=false

# sample filtering
min_io_delta=4  # samples with `len(input) - len(output) * min_io_ratio < min_io_delta` will be removed.

# config files
preprocess_config=conf/no_preprocess.yaml  # use conf/specaug.yaml for data augmentation
train_config=conf/train4gpu.yaml
lm_config=conf/lm.yaml
decode_config=conf/decode.yaml

# rnnlm related
use_wordlm=true     # false means to train/use a character LM
lm_vocabsize=65000  # effective only for word LMs
lm_resume=          # specify a snapshot file to resume LM training
lmtag=              # tag for managing LMs

# decoding parameter
n_average=10 # use 1 for RNN models
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'


datadir=/blob/v-jinx/data/WSJ_raw
datapredix=/var/storage/shared/msrmt/v-jinx/data/WSJ/espnet
exp_prefix=/blob/v-jinx/checkpoint_asr_nas
dumpdir=${datapredix}/${dumpdir}
# data
wsj0=${datadir}/csr_11
wsj1=${datadir}/csr_senn

# exp tag
tag="" # tag for managing experiments.


. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_si284
train_dev=test_dev93
train_test=test_eval92
recog_set="test_dev93 test_eval92"

echo "dumpdir is ${dumpdir}"

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}


dict=${datapredix}/data/lang_1char/${train_set}_units.txt
nlsyms=${datapredix}/data/lang_1char/non_lang_syms.txt

echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    ### Filter out short samples which lead to `loss_ctc=inf` during training
    ###  with the specified configuration.
    # Samples satisfying `len(input) - len(output) * min_io_ratio < min_io_delta` will be pruned.
    local/filtering_samples.py \
        --config ${train_config} \
        --preprocess-conf ${preprocess_config} \
        --data-json ${feat_tr_dir}/data.json \
        --mode-subsample "asr" \
        --arch-subsample "rnn" \
        ${min_io_delta:+--min-io-delta $min_io_delta} \
        --output-json-path ${feat_tr_dir}/data.json
fi