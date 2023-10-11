#!/usr/bin/env python3
#  2023, Sogang University;  Jeongkyun Park
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""1D-ResNet18."""

import contextlib
import logging
from typing import Tuple

import torch
import torch.nn as nn
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.conformer.swish import Swish
from espnet2.asr.frontend.abs_frontend import AbsFrontend


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding: int= 1, relu="relu"):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        if relu == "relu":
            self.relu = nn.ReLU(inpace=True)
        elif relu == "swish":
            self.relu = Swish()
        else:
            raise NotImplementedError


        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(identity)
        out = self.relu(out)
        return out


class ResNet18(AbsFrontend):
    def __init__(
        self, 
        in_channels: int, out_channels: int= 512, kernel_size: int= 80, 
        stride: int= 4, padding: int= 38, fs: int =  16000, relu_type: str= 'relu',
        freeze_weights: bool = False, pretrained_pt: str= None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu_type = relu_type
        if self.relu_type == "relu":
            self.relu = nn.ReLU(inpace=True)
        elif self.relu_type == "swish":
            self.relu = Swish()
        else:
            raise NotImplementedError

        self.layer1 = self._make_layer(64, 64, kernel_size=3, stride=1, num_blocks=2)
        self.layer2 = self._make_layer(64, 128, kernel_size=3, stride=2, num_blocks=2)
        self.layer3 = self._make_layer(128, 256, kernel_size=3, stride=2, num_blocks=2)
        self.layer4 = self._make_layer(256, out_channels, kernel_size=3, stride=2, num_blocks=2)

        self.avgpool = nn.AvgPool1d(kernel_size=20, stride=20)

        self.pretrained_pt = pretrained_pt

        self.freeze_weights = freeze_weights
        if self.freeze_weights: logging.info(f"{type(self)} states are frozen.")
        
    def reload_pretrained_parameters(self):
        if self.pretrained_pt is not None:
            states = torch.load(self.pretrained_pt, map_location='cpu')
            self.load_state_dict(states)
            logging.info(f"{type(self)} states successfully loaded from {self.pretrained_pt}")

    def _make_layer(self, in_channels, out_channels, kernel_size, stride, num_blocks):
        padding = kernel_size // 2
        layers = [BasicBlock(in_channels, out_channels, kernel_size, stride, padding=padding, relu=self.relu_type)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, kernel_size, stride=1, padding=padding, relu=self.relu_type))
        return nn.Sequential(*layers)

    def forward_length(self, input_lengths):
        input_lengths = cal_conv_length(self.conv1, input_lengths)
        
        for layers in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for layer in layers:
                input_lengths = cal_basicBlock_length(layer, input_lengths)
        
        input_lengths = cal_conv_length(self.avgpool, input_lengths)
        return input_lengths.to(int)

    def encode(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # (B, L, C) -> (B, C, L)
        assert input.dim() == 3, f"shape of the input is {input.shape}"
        
        input = input.permute(0,2,1)
        input = self.relu(self.bn1(self.conv1(input)))

        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)

        input = self.avgpool(input)
        # (B, D, L) -> (B, L, D)
        input = input.permute(0,2,1)
        input_lengths = self.forward_length(input_lengths)
        return input, input_lengths

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad() if self.freeze_weights else contextlib.nullcontext():
            feats, feats_lens = self.encode(input, input_lengths)
        return feats, feats_lens

    def output_size(self):
        return self.out_channels


def cal_basicBlock_length(module, lengths):
    lengths = cal_conv_length(module.conv1, lengths)
    lengths = cal_conv_length(module.conv2, lengths)
    return lengths

def cal_conv_length(module, lengths):
    return torch.floor((lengths + 2 * module.padding[0] - (module.kernel_size[0] - 1) - 1) / module.stride[0] + 1)