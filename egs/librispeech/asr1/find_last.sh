# general configuration
backend=pytorch
stage=-1       # start from -1 if you need to start from data download
stop_stage=100
ngpu=4         # number of gpus ("0" uses cpu, otherwise use gpu)
nj=32
debugmode=1
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option


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
tag="lh" # tag for managing experiments.



train_set=train_960
train_dev=dev
recog_set="test_clean"

dict=${datapredix}/data/lang_char/${train_set}_${bpemode}${nbpe}_units.txt
bpemodel=${datapredix}/data/lang_char/${train_set}_${bpemode}${nbpe}


expname=${train_set}_${backend}_${tag}
expdir=${exp_prefix}/exp/${expname}

#resume=${exp_prefix}/exp/train_960_pytorch_lh/results/model.loss.best       # Resume the training from snapshot
ls -l -trl ${expdir}/results | grep snapshot. | tail -1 >s.tmp
snap_num=$(awk -F ' ' '{print $NF}' s.tmp)
resume=${expdir}/results/${snap_num}

if [ -f ${resume} ]; then
   resume=${resume}
else
   resume=
fi

echo 'resume:'${resume}