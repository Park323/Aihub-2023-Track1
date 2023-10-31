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
    def __init__(self, input_size: int = 80, output_size: int = 384, model_path: str= None, freeze_weights: bool = False):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained("TheoJo/whisper-tiny-ko")
        self.encoder = AutoModelForSpeechSeq2Seq.from_pretrained("TheoJo/whisper-tiny-ko").model.encoder
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

            for idx, encoder_layer in enumerate(self.encoder.layers):
                if output_hidden_states:
                    encoder_states = encoder_states + (hidden_states,)
                # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
                dropout_probability = random.uniform(0, 1)
                if self.training and (dropout_probability < self.encoder.layerdrop):  # skip the layer
                    layer_outputs = (None, None)
                else:
                    if self.encoder.gradient_checkpointing and self.training:

                        def create_custom_forward(module):
                            def custom_forward(*inputs):
                                return module(*inputs, output_attentions)

                            return custom_forward

                        layer_outputs = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(encoder_layer),
                            hidden_states,
                            None,
                            None,
                        )
                    else:
                        layer_outputs = encoder_layer(
                            hidden_states,
                            None,
                            layer_head_mask=None,
                            output_attentions=output_attentions,
                        )

                    hidden_states = layer_outputs[0]

                if output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)

        hidden_states = self.encoder.layer_norm(hidden_states)

        ilens = (ilens / 2).ceil().to(int)

        return hidden_states, ilens, None