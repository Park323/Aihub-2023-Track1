# Copyright 2020 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Conformer encoder definition."""

import logging
import contextlib
from typing import List, Optional, Tuple, Union

import torch
from transformers import HubertModel, HubertConfig

from espnet2.asr.encoder.abs_encoder import AbsEncoder


class HuggingfaceHuBertEncoder(AbsEncoder):
    def __init__(self, input_size: int = 1, output_size: int = 768, model_path: str= None, freeze_weights: bool = False):
        super().__init__()
        config = HubertConfig.from_pretrained("team-lucid/hubert-base-korean")
        self.model = HubertModel(config)
        ckpt = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(ckpt)
        self.output_size_ = output_size
        
        self.freeze_weights = freeze_weights
        if self.freeze_weights: logging.info(f"{type(self)} states are frozen.")
    
    def output_size(self) -> int:
        return self.output_size_

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        xs_pad = xs_pad.permute(0,2,1) # (B, 1, T)
        
        output_attentions = self.model.config.output_attentions
        output_hidden_states = self.model.config.output_hidden_states

        with torch.no_grad() if self.freeze_weights else contextlib.nullcontext():
            feats_list = []
            for xs_pad_ in xs_pad:
                extract_feature = self.model.feature_extractor(xs_pad_)
                feats_list.append(extract_feature)
            extract_features = torch.cat(feats_list, dim=0)
            extract_features = extract_features.transpose(1, 2)
            
            # compute reduced length corresponding to feature vectors
            for kernel, stride in zip(self.model.config.conv_kernel, self.model.config.conv_stride):
                ilens = ((ilens - (kernel//2)) / stride).to(int)
            # attention mask
            mask_range = torch.range(1, extract_features.shape[1], dtype=int, device=extract_feature.device)
            mask_range = mask_range[None,:].repeat_interleave(extract_features.shape[0], dim=0)
            attention_mask = mask_range <= ilens[:, None]
            
            hidden_states = self.model.feature_projection(extract_features)
            hidden_states = self.model._mask_hidden_states(hidden_states)

            encoder_outputs = self.model.encoder(
                hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states
            )

        hidden_states = encoder_outputs[0]

        return hidden_states, ilens, None