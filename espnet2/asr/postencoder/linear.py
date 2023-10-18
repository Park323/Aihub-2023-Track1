from typeguard import check_argument_types

import torch
import torch.nn as nn
import torch.nn.functional as F

from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder


class SimpleLinear(AbsPostEncoder):
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
    ):
        super().__init__()
        self.input_size = input_size
        self._output_size = output_size
        self.linear1 = nn.Linear(self.input_size, self._output_size)

    def forward(
        self,
        inputs: torch.Tensor,
        input_lengths: torch.Tensor,
    ):
        assert check_argument_types()
        outputs = self.linear1(inputs)
        output_lengths = input_lengths

        return outputs, output_lengths
    
    def output_size(self) -> int:
        return self._output_size