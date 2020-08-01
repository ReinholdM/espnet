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

def count(word_frames, word_num, cut_degree):
    """

    :param phone_id_list: frame aligned from phone-frame.txt   1 1 1 35 37 28 1 1 26 26 1 1 1
    :param target_id_list: one sample of batches in ys_pad     35 37 28 26
    :return: half_length: the number of frames corresponding to a half length of target_id_list
    """
    cut_nums = int(word_num * cut_degree)
    cut_len = word_frames[cut_nums - 1]
    return cut_len


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


def get_mid_len_v2(y, frames, cut_degree):
    """

    :param torch.Tensor y: target y shape [B, Lmax]
    :return: mid_len: the list of the number of frames of a half of xs_pad shape [B]
    :return: wrd_nums: the list pf the number of word in this batch 
    """
    mid_len = [0 for x in range(y.shape[0])]
    wrd_nums = []
    #mid_len = []
    #with open("/data3/lhmeng/e2e-toolkit/espnet/espnet/nets/alignme/word2frame.txt", 'r') as f:
    #lines = f.readlines()
    """
    for i, line in enumerate(lines):
        seq = line.split()
        word_num = int(seq[0])
        frame_num = int(seq[1])
        word_frames = [int(i) for i in seq[2:(2 + word_num)]]
        char_id_list = [int(i) for i in seq[(2 + word_num):]]
        #logging.info("seq[1:] length " + str(seq[1:]))
    """
    for j in range(y.shape[0]):
        target = y[j,:].cpu().detach().numpy().tolist()
        target = list(filter(lambda x: x != -1, target))
        target = list(filter(lambda x: x != 2, target))
        #logging.info("y[j,:] "+str(target))
            #target = [x - 1 for x in target]
            #if dup_remov(seq[1:]) == y[j, :].detach().numpy().tolist():
            #s = [int(m) for m in dup_remov(seq[1:])]
        with open("/data3/lhmeng/e2e-toolkit/espnet/espnet/nets/alignme/word2frame.txt", 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                seq = line.split()
                word_num = int(seq[0])
                frame_num = int(seq[1])
                word_frames = [int(i) for i in seq[2:(2 + word_num)]]
                char_id_list = [int(i) for i in seq[(2 + word_num):]]
                #logging.info("seq[1:] length " + str(seq[1:]))
                if char_id_list == target and abs(frames[j] - frame_num) <= 3:
                    #seq.pop(0)
                    logging.info("frames[j]"+str(frames[j]))
                    logging.info("frames[j]"+str(frame_num))
                    logging.info("word_num "+str(j)+str(word_num))
                    mid_len[j] = (count(word_frames, word_num, cut_degree))
                    wrd_nums.append(word_num)
                    logging.info("char_id_list "+str(j)+ str(char_id_list))
                    logging.info("wrd_nums "+str(j)+' '+str(wrd_nums))
                    #mid_len.append(count(seq, target))
                    break
    return mid_len, wrd_nums
