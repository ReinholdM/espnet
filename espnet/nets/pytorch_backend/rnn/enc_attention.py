"""Attention modules for Output of Encoder."""

import math
import six
import logging

import torch
import torch.nn.functional as F

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.nets_utils import to_device

class AttDot(torch.nn.Module):
    """Dot product attention

    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int att_dim: attention dimension
    :param bool han_mode: flag to swith on mode of hierarchical attention and not store pre_compute_enc_h
    """

    def __init__(self, eprojs, dunits, att_dim, han_mode=False):
        super(AttDot, self).__init__()
        self.mlp_enc = torch.nn.Linear(eprojs, att_dim)
        self.mlp_dec = torch.nn.Linear(dunits, att_dim)

        self.dunits = dunits
        self.eprojs = eprojs
        self.att_dim = att_dim
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None
        self.han_mode = han_mode

    def reset(self):
        """reset states"""
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev, scaling=2.0):
        """AttDot forward

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_hs_len: padded encoder hidden state length (B)
        :param torch.Tensor dec_z: dummy (does not use)
        :param torch.Tensor att_prev: dummy (does not use)
        :param float scaling: scaling parameter before applying softmax
        :return: attention weighted encoder state (B, D_enc)
        :rtype: torch.Tensor
        :return: previous attention weight (B x T_max)
        :rtype: torch.Tensor
        """

        batch = enc_hs_pad.size(0)
        logging.info("batch ", batch)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None or self.han_mode:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_enc_h = torch.tanh(self.mlp_enc(self.enc_h))

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        e = torch.sum(self.pre_compute_enc_h * torch.tanh(self.mlp_dec(dec_z)).view(batch, 1, self.att_dim),
                      dim=2)  # utt x frame

        # NOTE consider zero padding when compute w.
        if self.mask is None:
            self.mask = to_device(self, make_pad_mask(enc_hs_len))
        e.masked_fill_(self.mask, -float('inf'))
        w = F.softmax(scaling * e, dim=1)

        # weighted sum over flames
        # utt x hdim
        # NOTE use bmm instead of sum(*)
        c = torch.sum(self.enc_h * w.view(batch, self.h_length, 1), dim=1)
        return c, w


def att_enc_for(args, num_att=1, han_mode=False):
    num_encs = getattr(args, "num_encs", 1)
    att_list = torch.nn.ModuleList()
    for i in range(num_att):
        att = initial_att('multi_head_dot', 320, 320, 320)
        att_list.append(att)

    return att_list

def initial_att(atype, eprojs, dunits, adim, han_mode=False):
    """Instantiates a single attention module"""
    if atype == 'multi_head_dot':
        att = AttDot(eprojs, dunits, adim, han_mode)
    return att
