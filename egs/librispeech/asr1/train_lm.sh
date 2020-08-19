#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=-1       # start from -1 if you need to start from data download
stop_stage=100
ngpu=4         # number of gpus ("0" uses cpu, otherwise use gpu)
nj=32
debugmode=1
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot

# feature configuration
do_delta=false

preprocess_config=conf/specaug.yaml
train_config=conf/train.yaml # current default recipe requires 4 gpus.
                             # if you do not have 4 gpus, please reconfigure the `batch-bins` and `accum-grad` parameters in config.
lm_config=conf/lm.yaml
decode_config=conf/decode.yaml

# rnnlm related
lm_resume= # specify a snapshot file to resume LM training
lmtag=     # tag for managing LMs

# decoding parameter
recog_model=model.acc.best  # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'
lang_model=rnnlm.model.best # set a language model to be used for decoding

# model average realted (only for transformer)
n_average=5                  # the number of ASR models to be averaged
use_valbest_average=true     # if true, the validation `n_average`-best ASR models will be averaged.
                             # if false, the last `n_average` ASR models will be averaged.
lm_n_average=0               # the number of languge models to be averaged
use_lm_valbest_average=false # if true, the validation `lm_n_average`-best language models will be averaged.
                             # if false, the last `lm_n_average` language models will be averaged.

# Set this to somewhere where you want to put your data, or where
# someone else has already put it.  You'll want to change this
# if you're not on the CLSP grid.
datapredix=/var/storage/shared/msrmt/v-jinx/data/LibriSpeech/espnet
dumpdir=${datapredix}/${dumpdir}
exp_prefix=/blob/v-jinx/checkpoint_asr_nas


# bpemode (unigram or bpe)
nbpe=5000
bpemode=unigram

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_960
train_dev=dev
recog_set="test_clean test_other dev_clean dev_other"

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}

lmexpname=train_rnnlm_${backend}_${lmtag}_${bpemode}${nbpe}_ngpu${ngpu}
lmexpdir=${exp_prefix}/exp/${lmexpname}
mkdir -p ${lmexpdir}

dict=${datapredix}/data/lang_char/${train_set}_${bpemode}${nbpe}_units.txt
bpemodel=${datapredix}/data/lang_char/${train_set}_${bpemode}${nbpe}


lmdatadir=${datapredix}/data/local/lm_train_${bpemode}${nbpe}
# use external data
if [ ! -e ${datapredix}/data/local/lm_train/librispeech-lm-norm.txt.gz ]; then
    wget http://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz -P ${datapredix}/data/local/lm_train/
fi
if [ ! -e ${lmdatadir} ]; then
    mkdir -p ${lmdatadir}
    cut -f 2- -d" " ${datapredix}/data/${train_set}/text | gzip -c > ${datapredix}/data/local/lm_train/${train_set}_text.gz
    # combine external text and transcriptions and shuffle them with seed 777
    zcat ${datapredix}/data/local/lm_train/librispeech-lm-norm.txt.gz ${datapredix}/data/local/lm_train/${train_set}_text.gz |\
        spm_encode --model=${bpemodel}.model --output_format=piece > ${lmdatadir}/train.txt
    cut -f 2- -d" " ${datapredix}/data/${train_dev}/text | spm_encode --model=${bpemodel}.model --output_format=piece \
                                                        > ${lmdatadir}/valid.txt
fi
${cuda_cmd} --gpu ${ngpu} ${lmexpdir}/train.log \
    lm_train.py \
    --config ${lm_config} \
    --ngpu ${ngpu} \
    --backend ${backend} \
    --verbose 1 \
    --outdir ${lmexpdir} \
    --tensorboard-dir tensorboard/${lmexpname} \
    --train-label ${lmdatadir}/train.txt \
    --valid-label ${lmdatadir}/valid.txt \
    --resume ${lm_resume} \
    --dict ${dict} \
    --dump-hdf5-path ${lmdatadir}