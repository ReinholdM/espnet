#!/bin/bash

# Copyright 2012  Microsoft Corporation  Johns Hopkins University (Author: Daniel Povey)
#           2015  Guoguo Chen
# Apache 2.0

# This script takes data prepared in a corpus-dependent way
# in data/local/, and converts it into the "canonical" form,
# in various subdirectories of data/, e.g. data/lang, data/lang_test_ug,
# data/train_si284, data/train_si84, etc.

# Don't bother doing train_si84 separately (although we have the file lists
# in data/local/) because it's just the first 7138 utterances in train_si284.
# We'll create train_si84 after doing the feature extraction.

lang_suffix=

echo "$0 $@"  # Print the command line for logging
. utils/parse_options.sh || exit 1;

. ./path.sh || exit 1;

datapredix=/var/storage/shared/msrmt/v-jinx/data/WSJ/espnet

echo "Preparing train and test data"
srcdir=data/local/data

for x in train_si284 test_eval92 test_eval93 test_dev93 test_eval92_5k test_eval93_5k test_dev93_5k dev_dt_05 dev_dt_20; do
  mkdir -p ${datapredix}/data/$x
  cp $srcdir/${x}_wav.scp ${datapredix}/data/$x/wav.scp || exit 1;
  cp $srcdir/$x.txt ${datapredix}/data/$x/text || exit 1;
  cp $srcdir/$x.spk2utt ${datapredix}/data/$x/spk2utt || exit 1;
  cp $srcdir/$x.utt2spk ${datapredix}/data/$x/utt2spk || exit 1;
  utils/filter_scp.pl ${datapredix}/data/$x/spk2utt $srcdir/spk2gender > ${datapredix}/data/$x/spk2gender || exit 1;
done

echo "Succeeded in formatting data."
