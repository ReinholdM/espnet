export LANG=en_US.UTF-8
export LANGUAGE=
export LC_CTYPE="en_US.UTF-8"
export LC_NUMERIC=zh_CN.UTF-8
export LC_TIME=zh_CN.UTF-8
export LC_COLLATE="en_US.UTF-8"
export LC_MONETARY=zh_CN.UTF-8
export LC_MESSAGES="en_US.UTF-8"
export LC_PAPER=zh_CN.UTF-8
export LC_NAME=zh_CN.UTF-8
export LC_ADDRESS=zh_CN.UTF-8
export LC_TELEPHONE=zh_CN.UTF-8
export LC_MEASUREMENT=zh_CN.UTF-8
export LC_IDENTIFICATION=zh_CN.UTF-8
export LC_ALL=

sudo apt-get update
sudo apt-get install -y locales
sudo locale-gen en_US.UTF-8
sudo locale-gen zh_CN.UTF-8

sudo apt-get update
sudo apt-get install -y locales

PHILLY_USER=xuta
version=1.2.0
path=/opt/conda/envs/pytorch-py3.6/bin:/opt/conda/bin:

sudo rm /etc/sudoers.d/${PHILLY_USER}
sudo touch /etc/sudoers.d/${PHILLY_USER}
sudo chmod 777 /etc/sudoers.d/${PHILLY_USER}
sudo echo "Defaults        secure_path=\"$path:/usr/local/mpi/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\"" > /etc/sudoers.d/${PHILLY_USER}
sudo chmod 0440 /etc/sudoers.d/${PHILLY_USER}

echo 'Now into the script'

cd /var/storage/shared/msrmt/v-jinx/lh/espnet/tools
ln -sf /home/espnet/tools/* ./

echo 'finish link'

cd /var/storage/shared/msrmt/v-jinx/lh/espnet/egs/librispeech/asr1

pwd

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

# feature configuration
do_delta=false
preprocess_config=  # use conf/specaug.yaml for data augmentation
train_config=conf/train_prob_mix.yaml # current default recipe requires 4 gpus.
                             # if you do not have 4 gpus, please reconfigure the `batch-bins` and `accum-grad` parameters in config.
lm_config=conf/lm.yaml
decode_config=conf/decode_transformer.yaml

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
dumpdir=${datapredix}
exp_prefix=/blob/v-jinx/checkpoint_lh_asr


# bpemode (unigram or bpe)
nbpe=5000
bpemode=unigram

# exp tag
tag="lh_mix01" # tag for managing experiments.


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


#resume=${exp_prefix}/exp/train_960_pytorch_lh/results/model.loss.best       # Resume the training from snapshot
if [ -f ${expdir}/results ]; then

    ls -l -trl ${expdir}/results | grep snapshot. | tail -1 >s.tmp
    snap_num=$(awk -F ' ' '{print $NF}' s.tmp)
    resume=${expdir}/results/${snap_num}
    if [ -f ${resume} ]; then
       resume=${resume}
    else
       resume=
    fi
else
    resume=
fi

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
    asr_train.py \
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
