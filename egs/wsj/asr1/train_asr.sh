
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
lmtag=lm              # tag for managing LMs

# decoding parameter
n_average=10 # use 1 for RNN models
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'


datadir=/blob/v-jinx/data/WSJ_raw
datapredix=/var/storage/shared/msrmt/v-jinx/data/WSJ/espnet
exp_prefix=/blob/v-jinx/checkpoint_asr_nas/wsj
dumpdir=${datapredix}/${dumpdir}

# exp tag
tag="wsj_asr" # tag for managing experiments.

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_si284
train_dev=test_dev93
train_test=test_eval92
recog_set="test_dev93 test_eval92"



feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}


dict=${datapredix}/data/lang_1char/${train_set}_units.txt
nlsyms=${datapredix}/data/lang_1char/non_lang_syms.txt


lmexpname=train_rnnlm_${backend}_${lmtag}
lmexpdir=${exp_prefix}/exp/${lmexpname}
mkdir -p ${lmexpdir}

if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})_$(basename ${preprocess_config%.*})
    if ${do_delta}; then
        expname=${expname}_delta
    fi
else
    expname=${train_set}_${backend}_${tag}
fi
expdir=${exp_prefix}/exp/${expname}
mkdir -p ${expdir}


#python ../../../espnet/bin/asr_train.py \
#--config ${train_config} \
#--preprocess-conf ${preprocess_config} \
#--ngpu ${ngpu} \
#--backend ${backend} \
#--outdir ${expdir}/results \
#--tensorboard-dir tensorboard/${expname} \
#--debugmode ${debugmode} \
#--dict ${dict} \
#--debugdir ${expdir} \
#--minibatches ${N} \
#--verbose ${verbose} \
#--resume ${resume} \
#--seed ${seed} \
#--train-json ${feat_tr_dir}/data.json \
#--valid-json ${feat_dt_dir}/data.json

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Network Training"

    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train.py \
        --config ${train_config} \
        --preprocess-conf ${preprocess_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/${expname} \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --seed ${seed} \
        --train-json ${feat_tr_dir}/data.json \
        --valid-json ${feat_dt_dir}/data.json
fi


