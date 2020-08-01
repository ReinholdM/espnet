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

# Set this to somewhere where you want to put your data, or where
# someone else has already put it.  You'll want to change this
# if you're not on the CLSP grid.
datapredix=/var/storage/shared/msrmt/v-jinx/data/LibriSpeech/espnet
dumpdir=${datapredix}/${dumpdir}
exp_prefix=/blob/v-jinx/checkpoint_lh_asr


# bpemode (unigram or bpe)
nbpe=5000
bpemode=unigram

# exp tag
tag="lh" # tag for managing experiments.


# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_960
train_dev=dev
recog_set="test_clean"


feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}

dict=${datapredix}/data/lang_char/${train_set}_${bpemode}${nbpe}_units.txt
bpemodel=${datapredix}/data/lang_char/${train_set}_${bpemode}${nbpe}


expname=${train_set}_${backend}_${tag}
expdir=${exp_prefix}/exp/${expname}
mkdir -p ${expdir}

echo 'exp_dir'
echo ${expdir}

echo 'begin_run'

echo ${ngpu}
echo ${expdir}
echo ${train_config}
echo ${backend}
echo ${expdir}
echo ${exp_prefix}
echo ${debugmode}
echo ${expdir}
echo ${dict}
echo ${verbose}
echo ${resume}
echo ${bpemode}
echo ${nbpe}
echo ${feat_tr_dir}
echo ${feat_dt_dir}

${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
    asr_train_custom.py \
    --config ${train_config} \
    --preprocess-conf ${preprocess_config} \
    --ngpu ${ngpu} \
    --backend ${backend} \
    --outdir ${expdir}/results \
    --tensorboard-dir ${exp_prefix}/tensorboard/${expname} \
    --debugmode ${debugmode} \
    --dict ${dict} \
    --debugdir ${expdir} \
    --minibatches ${N} \
    --verbose ${verbose} \
    --resume ${resume} \
    --train-json ${feat_tr_dir}/data_${bpemode}${nbpe}.json \
    --valid-json ${feat_dt_dir}/data_${bpemode}${nbpe}.json

