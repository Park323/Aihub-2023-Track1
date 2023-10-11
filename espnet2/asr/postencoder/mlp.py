from abc import ABC, abstractmethod
import random
from typing import Dict, List, Optional, Tuple, Union
from typeguard import check_argument_types

import torch
import torch.nn as nn
import torch.nn.functional as F

from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder


class SimpleMLP(AbsPostEncoder):
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        expand_ratio: int = 4,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.input_size = input_size
        self._output_size = output_size
        self.linear1 = nn.Linear(self.input_size, self._output_size*expand_ratio)
        self.bn = nn.BatchNorm1d(self._output_size*expand_ratio)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(self._output_size*expand_ratio, output_size)

    def forward(
        self,
        inputs: torch.Tensor,
        input_lengths: torch.Tensor,
    ):
        assert check_argument_types()
        outputs = self.linear1(inputs).permute(0,2,1)
        outputs = F.relu(self.bn(outputs)).permute(0,2,1)
        outputs = self.dropout(outputs)
        outputs = self.linear2(outputs)
        output_lengths = input_lengths

        return outputs, output_lengths
    
    def output_size(self) -> int:
        return self._output_size