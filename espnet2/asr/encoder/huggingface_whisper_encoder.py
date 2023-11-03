# Copyright 2020 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Conformer encoder definition."""

import random
import logging
import contextlib
from typing import List, Optional, Tuple, Union

import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

from espnet2.asr.encoder.abs_encoder import AbsEncoder


class HuggingfaceWhisperEncoder(AbsEncoder):
    def __init__(self, input_size: int = 80, output_size: int = 384, n_layer:int = 12, 
                 model_path: str="TheoJo/whisper-tiny-ko", freeze_weights: bool = False,
                 interctc_layer_idx: List[int] = [], interctc_use_conditioning: bool = False):
        super().__init__()
        self.encoder = AutoModelForSpeechSeq2Seq.from_pretrained(model_path).model.encoder
        self.encoder.layers = self.encoder.layers[:n_layer]
        
        self.output_size_ = output_size
        
        self.freeze_weights = freeze_weights
        if self.freeze_weights: logging.info(f"{type(self)} states are frozen.")
        
        self.interctc_layer_idx = interctc_layer_idx
        if len(interctc_layer_idx) > 0:
            assert 0 < min(interctc_layer_idx) and max(interctc_layer_idx) < n_layer
        self.interctc_use_conditioning = interctc_use_conditioning
    
    def output_size(self) -> int:
        return self.output_size_

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
        ctc: CTC = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        xs_pad = xs_pad.permute(0,2,1) # (B, 80, T)
        
        output_attentions = self.encoder.config.output_attentions
        output_hidden_states = self.encoder.config.output_hidden_states
        
        with torch.no_grad() if self.freeze_weights else contextlib.nullcontext():
            inputs_embeds = torch.nn.functional.gelu(self.encoder.conv1(xs_pad))
            inputs_embeds = torch.nn.functional.gelu(self.encoder.conv2(inputs_embeds))

            inputs_embeds = inputs_embeds.permute(0, 2, 1)
            embed_pos = self.encoder.embed_positions.weight

            hidden_states = inputs_embeds + embed_pos[:inputs_embeds.shape[1]]
            hidden_states = torch.nn.functional.dropout(hidden_states, p=self.encoder.dropout, training=self.training)

            encoder_states = () if output_hidden_states else None
            all_attentions = () if output_attentions else None

            intermediate_outs = []
            for idx, encoder_layer in enumerate(self.encoder.layers, start=1):
                if output_hidden_states:
                    encoder_states = encoder_states + (hidden_states,)
                # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
                dropout_probability = random.uniform(0, 1)
                
                layer_outputs = encoder_layer(
                    hidden_states,
                    None,
                    layer_head_mask=None,
                    output_attentions=output_attentions,
                )

                hidden_states = layer_outputs[0]
                
                if idx in self.interctc_layer_idx:
                    encoder_out = hidden_states

                    # intermediate outputs are also normalized
                    encoder_out = self.encoder.layer_norm(encoder_out)

                    intermediate_outs.append((idx, encoder_out))

                    # if self.interctc_use_conditioning:
                    #     ctc_out = ctc.softmax(encoder_out)

                    #     xs_pad = xs_pad + self.conditioning_layer(ctc_out)

                if output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)

        hidden_states = self.encoder.layer_norm(hidden_states)

        olens = (ilens / 2).ceil().to(int)

        if intermediate_outs:
            return (hidden_states, intermediate_outs), olens, None
        return hidden_states, olens, None