#!/bin/sh

#  avg_hidden_states.py
#
#
#  Created by Meng Linghui on 5/21/20.
# This script filters out words which are not in our corpus.
# It requires a file represent alignment between frame and phones: phone-frame.txt

import os
import sys
import torch
import torch.nn  as nn
import numpy as np
import logging
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from espnet.nets.pytorch_backend.nets_utils import pad_list


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
    # logging.info("results "+str(results))
    return results


def get_mid_len_v2(y, frames):
    """

    :param torch.Tensor y: target y shape [B, Lmax]
    :param frames: the list of frame number in each entry
    :return: total_frames: the list of each entry with word_frames [B, word_nums] [[2,5,19],[2,10,32],[12,45,222]]
    """
    mid_len = [0 for x in range(y.shape[0])]
    wrd_nums = []

    total_frames = []
    for j in range(y.shape[0]):
        target = y[j, :].cpu().detach().numpy().tolist()
        target = list(filter(lambda x: x != -1, target))
        target = list(filter(lambda x: x != 2, target))
        # logging.info("y[j,:] "+str(target))
        # target = [x - 1 for x in target]
        # if dup_remov(seq[1:]) == y[j, :].detach().numpy().tolist():
        # s = [int(m) for m in dup_remov(seq[1:])]
        with open("/data3/lhmeng/e2e-toolkit/espnet/espnet/nets/alignme/word2frame.txt", 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                seq = line.split()
                word_num = int(seq[0])
                frame_num = int(seq[1])
                word_frames = [int(i) for i in seq[2:(2 + word_num)]]
                word_frames.append(frame_num)
                char_id_list = [int(i) for i in seq[(2 + word_num):]]
                # logging.info("seq[1:] length " + str(seq[1:]))
                if char_id_list == target and abs(frames[j] - frame_num) <= 1:
                    total_frames.append(word_frames)
                    break
    return total_frames


def avg_hs(hs_pad, hlens, total_frames):
    """

    :param hs_pad: the encoder output (B, Tmax, hdim)
    :param hlens: hidden state length B
    :param total_frames: list (B, word_frames)
    :return: hs_avg_pad
    """
    logging.info("hs_pad shape", hs_pad.shape)
    #logging.info("hs_pad", hs_pad)
    #hs_pack = pack_padded_sequence(hs_pad, hlens, batch_first=True)
    hs_pack = []
    for x in hs_pad:
        hs_pack.append([s[s!=0] for s in x])
        #hs_pack.append(s[s!=0] for s in x)
    #hs_pack = torch.from_numpy(np.array(hs_pack))
    #hs_pack = [x[x!=0.0000e+00] for x in hs_pad]
    #logging.info("hs_pack",hs_pack)
    hs_avg_pack = []
    avg_len = []
    for i, frame_nums in enumerate(total_frames):
        frame_nums.insert(0, 0)
        for x in range(len(frame_nums)):
            frame_nums[x] = frame_nums[x] // 4
        logging.info("hs_pack[i][0].dtype", hs_pack[i][0])
        logging.info("frame_nums", frame_nums)
        #hs_pack[i] = torch.from_numpy(np.array(hs_pack[i])).unsqueeze(1)
        avg = []
        for j in range(len(frame_nums)-1):
            #avg += [torch.sum(hs_pack[i][frame_nums[j]:frame_nums[j + 1]], 0).unsqueeze_(0) / (
            #        frame_nums[j + 1] - frame_nums[j])]
            #avg_mid = [torch.from_numpy(np.sum([hs_pack[i][x] for x in range(frame_nums[j], frame_nums[j+1])], axis=0))]
            logging.info("hs_pack[i][0]", len(hs_pack[i][0]))
            logging.info("hs_pack[i][0].dtype", hs_pack[i][0])
            avg_mid = np.sum([hs_pack[i][x] for x in range(frame_nums[j], frame_nums[j+1])], axis=0)
            '''
            for m in range(avg_mid):
                m = m / (frame_nums[j+1] -frame_nums[j])
            '''
            avg_mid = avg_mid / (frame_nums[j+1] -frame_nums[j])
            logging.info("avg_mid",avg_mid)
            logging.info("avg_mid size", avg_mid.size())
            #avg_mid = torch.from_numpy(avg_mid)
            #logging.info("avg_mid size", avg_mid.size)
            avg.append(avg_mid)
            #logging.info("len(avg_mid)", len(avg_mid))
        logging.info("len(avg)",len(avg))
        avg_len.append(len(avg))
        #hs_avg_pack.append(torch.cat(avg, dim=0))
        hs_avg_pack.append(avg)
    logging.info("len(hs_avg_pack)", len(hs_avg_pack))
    logging.info("avg_len",avg_len)
    #hs_avg_pad = pad_packed_sequence(hs_avg_pack, batch_first=True)
    hs_avg_pad = pad_list(hs_avg_pack, 0)
    logging.info("hs_avg_pad", hs_avg_pad)
    logging.info("hs_avg_pad shape", hs_avg_pad.shape)
    return hs_avg_pad, avg_len

