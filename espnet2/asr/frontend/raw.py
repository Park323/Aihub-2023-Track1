from typing import Tuple
import torch
from espnet2.asr.frontend.abs_frontend import AbsFrontend


class RawFrontend(AbsFrontend):
    """Use raw waveform for ASR."""

    def __init__(self, n_channel:int=1, **kwargs):
        super().__init__()
        self.n_channel = n_channel
        
    def output_size(self) -> int:
        return self.n_channel

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return input, input_lengths
