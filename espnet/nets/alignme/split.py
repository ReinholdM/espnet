#!/bin/sh

#  split.py
#
#
#  Created by Meng Linghui on 4/18/20.
# This script filters out words which are not in our corpus.
# It requires a file represent alignment between frame and phones: phone-frame.txt

import os
import sys
import torch
import torch.nn  as nn
import numpy as np
import logging

def count(phone_id_list, target_id_list):
    """

    :param phone_id_list: frame aligned from phone-frame.txt   1 1 1 35 37 28 1 1 26 26 1 1 1
    :param target_id_list: one sample of batches in ys_pad     35 37 28 26
    :return: half_length: the number of frames corresponding to a half length of target_id_list
    """
    phone_id_list = [int(x) for x in phone_id_list]
    cnt = 0
    mid_len = int(len(target_id_list) / 2)
    pointer = mid_len
    target_id_list = target_id_list[:mid_len]
    for i in range(len(phone_id_list)):
        if phone_id_list[i] == 1:
            cnt += 1
        if phone_id_list[i] == target_id_list[0]:
            cnt += 1
            if phone_id_list[i] != phone_id_list[i - 1]:
                target_id_list.pop(0)

        if len(target_id_list) == 0:
            break
    return cnt


def dup_remov(phone_id_list):
    """

    :param torch.Tensor phone_id_list: [B]
    :return: list results: After removing duplicated list shape [B]
    """
    results = phone_id_list
    for i in range(1, len(phone_id_list)):
        if phone_id_list[i] == phone_id_list[i - 1]:
            results[i - 1] = '0'
    results = list(filter(lambda x: x != '0', results))
    results = list(filter(lambda x: x != '1', results))
    #logging.info("results "+str(results))
    return results


def get_mid_len_v1(y, utt_id_list):
    """

    :param y: target y shape [B, Lmax]
    :param utt_id_list: the list of uttid corresponding to the target y shape [B]
    :return: mid_len: the list of the number of frames of a half of xs_pad shape [B]
    """
    mid_len = [0 for x in range(y.shape[0])]
    with open("phone-frame.txt") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            seq = line.split()
            for uttid in utt_id_list:
                if uttid == seq[0]:
                    seq.pop(0)
                    mid_len[i] = (count(seq, y[i, :]))
    return mid_len


def get_mid_len_v2(y):
    """

    :param torch.Tensor y: target y shape [B, Lmax]
    :return: mid_len: the list of the number of frames of a half of xs_pad shape [B]
    """
    mid_len = [0 for x in range(y.shape[0])]
    #mid_len = []
    with open("/data3/lhmeng/e2e-toolkit/espnet/espnet/nets/alignme/train.phone.frame", 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            seq = line.split()
            #logging.info("seq[1:] length " + str(seq[1:]))
            for j in range(y.shape[0]):
                target = y[j,:].cpu().detach().numpy().tolist()
                #logging.info("ys_pad[j,:] "+str(target))
                target = list(filter(lambda x: x != -1, target))
                target = list(filter(lambda x: x != 2, target))
                target = [x - 1 for x in target]
                #if dup_remov(seq[1:]) == y[j, :].detach().numpy().tolist():
                s = [int(m) for m in dup_remov(seq[1:])]
                if s == target:
                    seq.pop(0)
                    mid_len[j] = (count(seq, target))
                    #mid_len.append(count(seq, target))
    return mid_len
