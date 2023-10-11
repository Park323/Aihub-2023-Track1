"""Augment module."""
from typing import Any, Optional, Sequence, Union

import torch
import numpy as np
from scipy import signal

from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.layers.mask_along_axis import MaskAlongAxis, MaskAlongAxisVariableMaxWidth


class TimeAug(AbsSpecAug):
    """Implementations of Augmentations for raw waveform.

    Reference:
        E. Kharitonov et al.
        "Data augmenting contrastive learning of speech representations in the time domain"

    .. warning::
    """

    def __init__(
        self,
        samplerate:int = 16000,
        apply_bandrej: bool = True,
        bandwidth_range: Optional[Union[int, Sequence[int]]] = None,
        num_bandrej: int = 2,
        apply_tdrop: bool = True,
        time_mask_width_range: Optional[Union[int, Sequence[int]]] = None,
        time_mask_width_ratio_range: Optional[Union[float, Sequence[float]]] = None,
        num_time_mask: int = 2,
    ):
        if not apply_bandrej and not apply_tdrop:
            raise ValueError(
                "Neither one of tdrop nor bandrej is being applied"
            )
        if (
            apply_tdrop
            and (time_mask_width_range is not None)
            and (time_mask_width_ratio_range is not None)
        ):
            raise ValueError(
                'Either one of "time_mask_width_range" or '
                '"time_mask_width_ratio_range" can be used'
            )
        super().__init__()
        self.apply_tdrop = apply_tdrop
        self.apply_bandrej = apply_bandrej

        if apply_tdrop:
            if time_mask_width_range is not None:
                self.time_mask = MaskAlongAxis(
                    dim="time",
                    mask_width_range=time_mask_width_range,
                    num_mask=num_time_mask,
                )
            elif time_mask_width_ratio_range is not None:
                self.time_mask = MaskAlongAxisVariableMaxWidth(
                    dim="time",
                    mask_width_ratio_range=time_mask_width_ratio_range,
                    num_mask=num_time_mask,
                )
            else:
                raise ValueError(
                    'Either one of "time_mask_width_range" or '
                    '"time_mask_width_ratio_range" should be used.'
                )
        else:
            self.time_mask = None

        if apply_bandrej:
            self.bandrej = BandRejectFilter(
                fs=samplerate,
                bandwidth_range=bandwidth_range,
                num_mask=num_bandrej,
            )
        else:
            self.bandrej = None

    def forward(self, x, x_lengths=None):
        if self.bandrej is not None:
            new_x = []
            for signal in x.cpu().detach().numpy():
                new_x.append(torch.tensor(self.bandrej(signal)))
            x = torch.stack(new_x).to(x.dtype).to(x.device)
        if self.time_mask is not None:
            x, x_lengths = self.time_mask(x, x_lengths)
        return x, x_lengths


class BandRejectFilter:
    def __init__(
        self,
        bandwidth_range:Union[int, Sequence[float]], 
        fs:int=16000, num_mask:int=2, normalized:bool=False,
    ) -> None:
        """ fs : Sample frequency (Hz)"""
        self.fs = fs
        self.num_mask = num_mask
        self.bandwidth_range = bandwidth_range
        self.normalized = normalized

    def __call__(self, sig) -> Any:
        """ f0 : Frequency to be removed from signal (Hz)
            Q  : Quality factor = f0/bandwidth"""
        f0s = [np.random.random() * 8000 for _ in range(self.num_mask)]
        bandwidth = self.bandwidth_range if isinstance(self.bandwidth_range, int) else np.random.uniform(*self.bandwidth_range)
        filtered_sig = sig
        for f0 in f0s:
            f0 = f0/(self.fs/2) if self.normalized else f0
            b_i, a_i = signal.iirnotch(f0, f0/bandwidth, self.fs)
            filtered_sig = signal.lfilter(b_i, a_i, filtered_sig)
        return filtered_sig