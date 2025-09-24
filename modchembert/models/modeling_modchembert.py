# Copyright 2025 Emmanuel Cortes, All Rights Reserved.
#
# Copyright 2024 Answer.AI, LightOn, and contributors, and the HuggingFace Inc. team. All rights reserved.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file is adapted from the transformers library.
# Modifications include:
# - Additional classifier_pooling options for ModChemBertForSequenceClassification
#   - sum_mean, sum_sum, mean_sum, mean_mean: from ChemLM (utilizes all hidden states)
#   - max_cls, cls_mha, max_seq_mha: from MaxPoolBERT (utilizes last k hidden states)
#   - max_seq_mean: a merge between sum_mean and max_cls (utilizes last k hidden states)
# - Addition of ModChemBertPoolingAttention for cls_mha and max_seq_mha pooling options

import copy
import math
import typing
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from transformers.modeling_outputs import MaskedLMOutput, SequenceClassifierOutput
from transformers.models.modernbert.modeling_modernbert import (
    MODERNBERT_ATTENTION_FUNCTION,
    ModernBertModel,
    ModernBertPredictionHead,
    ModernBertPreTrainedModel,
    ModernBertRotaryEmbedding,
    _pad_modernbert_output,
    _unpad_modernbert_input,
)
from transformers.utils import logging

from .configuration_modchembert import ModChemBertConfig

logger = logging.get_logger(__name__)


class InitWeightsMixin:
    def _init_weights(self, module: nn.Module):
        super()._init_weights(module)  # type: ignore

        cutoff_factor = self.config.initializer_cutoff_factor  # type: ignore
        if cutoff_factor is None:
            cutoff_factor = 3

        def init_weight(module: nn.Module, std: float):
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(
                    module.weight,
                    mean=0.0,
                    std=std,
                    a=-cutoff_factor * std,
                    b=cutoff_factor * std,
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        stds = {
            "in": self.config.initializer_range,  # type: ignore
            "out": self.config.initializer_range / math.sqrt(2.0 * self.config.num_hidden_layers),  # type: ignore
            "final_out": self.config.hidden_size**-0.5,  # type: ignore
        }

        if isinstance(module, ModChemBertForMaskedLM):
            init_weight(module.decoder, stds["out"])
        elif isinstance(module, ModChemBertForSequenceClassification):
            init_weight(module.classifier, stds["final_out"])
        elif isinstance(module, ModChemBertPoolingAttention):
            init_weight(module.Wq, stds["in"])
            init_weight(module.Wk, stds["in"])
            init_weight(module.Wv, stds["in"])
            init_weight(module.Wo, stds["out"])


class ModChemBertPoolingAttention(nn.Module):
    """Performs multi-headed self attention on a batch of sequences."""

    def __init__(self, config: ModChemBertConfig):
        super().__init__()
        self.config = copy.deepcopy(config)
        # Override num_attention_heads to use classifier_pooling_num_attention_heads
        self.config.num_attention_heads = config.classifier_pooling_num_attention_heads
        # Override attention_dropout to use classifier_pooling_attention_dropout
        self.config.attention_dropout = config.classifier_pooling_attention_dropout

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention heads "
                f"({config.num_attention_heads})"
            )

        self.attention_dropout = config.attention_dropout
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.head_dim * self.num_heads
        self.Wq = nn.Linear(config.hidden_size, self.all_head_size, bias=config.attention_bias)
        self.Wk = nn.Linear(config.hidden_size, self.all_head_size, bias=config.attention_bias)
        self.Wv = nn.Linear(config.hidden_size, self.all_head_size, bias=config.attention_bias)

        # Use global attention
        self.local_attention = (-1, -1)
        rope_theta = config.global_rope_theta
        # sdpa path from original ModernBert implementation
        config_copy = copy.deepcopy(config)
        config_copy.rope_theta = rope_theta
        self.rotary_emb = ModernBertRotaryEmbedding(config=config_copy)

        self.Wo = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.out_drop = nn.Dropout(config.attention_dropout) if config.attention_dropout > 0.0 else nn.Identity()
        self.pruned_heads = set()

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        bs, seq_len = kv.shape[:2]
        q_proj: torch.Tensor = self.Wq(q)
        k_proj: torch.Tensor = self.Wk(kv)
        v_proj: torch.Tensor = self.Wv(kv)
        qkv = torch.stack(
            (
                q_proj.reshape(bs, seq_len, self.num_heads, self.head_dim),
                k_proj.reshape(bs, seq_len, self.num_heads, self.head_dim),
                v_proj.reshape(bs, seq_len, self.num_heads, self.head_dim),
            ),
            dim=2,
        )  # (bs, seq_len, 3, num_heads, head_dim)

        device = kv.device
        if attention_mask is None:
            attention_mask = torch.ones((bs, seq_len), device=device, dtype=torch.bool)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).long()

        attn_outputs = MODERNBERT_ATTENTION_FUNCTION["sdpa"](
            self,
            qkv=qkv,
            attention_mask=_prepare_4d_attention_mask(attention_mask, kv.dtype),
            sliding_window_mask=None,  # not needed when using global attention
            position_ids=position_ids,
            local_attention=self.local_attention,
            bs=bs,
            dim=self.all_head_size,
            **kwargs,
        )
        hidden_states = attn_outputs[0]
        hidden_states = self.out_drop(self.Wo(hidden_states))

        return hidden_states


class ModChemBertForMaskedLM(InitWeightsMixin, ModernBertPreTrainedModel):
    config_class = ModChemBertConfig
    _tied_weights_keys = ["decoder.weight"]

    def __init__(self, config: ModChemBertConfig):
        super().__init__(config)
        self.config = config
        self.model = ModernBertModel(config)
        self.head = ModernBertPredictionHead(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=config.decoder_bias)

        self.sparse_prediction = self.config.sparse_prediction
        self.sparse_pred_ignore_index = self.config.sparse_pred_ignore_index

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.decoder

    def set_output_embeddings(self, new_embeddings: nn.Linear):
        self.decoder = new_embeddings

    @torch.compile(dynamic=True)
    def compiled_head(self, output: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.head(output))

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        sliding_window_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        indices: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        batch_size: int | None = None,
        seq_len: int | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor] | tuple[torch.Tensor, typing.Any] | MaskedLMOutput:
        r"""
        sliding_window_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding or far-away tokens. In ModernBert, only every few layers
            perform global attention, while the rest perform local attention. This mask is used to avoid attending to
            far-away tokens in the local attention layers when not using Flash Attention.
        indices (`torch.Tensor` of shape `(total_unpadded_tokens,)`, *optional*):
            Indices of the non-padding tokens in the input sequence. Used for unpadding the output.
        cu_seqlens (`torch.Tensor` of shape `(batch + 1,)`, *optional*):
            Cumulative sequence lengths of the input sequences. Used to index the unpadded tensors.
        max_seqlen (`int`, *optional*):
            Maximum sequence length in the batch excluding padding tokens. Used to unpad input_ids & pad output tensors.
        batch_size (`int`, *optional*):
            Batch size of the input sequences. Used to pad the output tensors.
        seq_len (`int`, *optional*):
            Sequence length of the input sequences including padding tokens. Used to pad the output tensors.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        self._maybe_set_compile()

        if self.config._attn_implementation == "flash_attention_2":  # noqa: SIM102
            if indices is None and cu_seqlens is None and max_seqlen is None:
                if batch_size is None and seq_len is None:
                    if inputs_embeds is not None:
                        batch_size, seq_len = inputs_embeds.shape[:2]
                    else:
                        batch_size, seq_len = input_ids.shape[:2]  # type: ignore
                device = input_ids.device if input_ids is not None else inputs_embeds.device  # type: ignore

                if attention_mask is None:
                    attention_mask = torch.ones((batch_size, seq_len), device=device, dtype=torch.bool)  # type: ignore

                if inputs_embeds is None:
                    with torch.no_grad():
                        input_ids, indices, cu_seqlens, max_seqlen, position_ids, labels = _unpad_modernbert_input(
                            inputs=input_ids,  # type: ignore
                            attention_mask=attention_mask,  # type: ignore
                            position_ids=position_ids,
                            labels=labels,
                        )
                else:
                    inputs_embeds, indices, cu_seqlens, max_seqlen, position_ids, labels = _unpad_modernbert_input(
                        inputs=inputs_embeds,
                        attention_mask=attention_mask,  # type: ignore
                        position_ids=position_ids,
                        labels=labels,
                    )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            indices=indices,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            batch_size=batch_size,
            seq_len=seq_len,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = outputs[0]

        if self.sparse_prediction and labels is not None:
            # flatten labels and output first
            labels = labels.view(-1)
            last_hidden_state = last_hidden_state.view(labels.shape[0], -1)

            # then filter out the non-masked tokens
            mask_tokens = labels != self.sparse_pred_ignore_index
            last_hidden_state = last_hidden_state[mask_tokens]
            labels = labels[mask_tokens]

        logits = (
            self.compiled_head(last_hidden_state)
            if self.config.reference_compile
            else self.decoder(self.head(last_hidden_state))
        )

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, vocab_size=self.config.vocab_size, **kwargs)

        if self.config._attn_implementation == "flash_attention_2":
            with nullcontext() if self.config.repad_logits_with_grad or labels is None else torch.no_grad():
                logits = _pad_modernbert_output(inputs=logits, indices=indices, batch=batch_size, seqlen=seq_len)  # type: ignore

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return MaskedLMOutput(
            loss=loss,
            logits=typing.cast(torch.FloatTensor, logits),
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class ModChemBertForSequenceClassification(InitWeightsMixin, ModernBertPreTrainedModel):
    config_class = ModChemBertConfig

    def __init__(self, config: ModChemBertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.model = ModernBertModel(config)
        if self.config.classifier_pooling in {"cls_mha", "max_seq_mha"}:
            self.pooling_attn = ModChemBertPoolingAttention(config=self.config)
        else:
            self.pooling_attn = None
        self.head = ModernBertPredictionHead(config)
        self.drop = torch.nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        sliding_window_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        indices: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        batch_size: int | None = None,
        seq_len: int | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor] | tuple[torch.Tensor, typing.Any] | SequenceClassifierOutput:
        r"""
        sliding_window_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding or far-away tokens. In ModernBert, only every few layers
            perform global attention, while the rest perform local attention. This mask is used to avoid attending to
            far-away tokens in the local attention layers when not using Flash Attention.
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        indices (`torch.Tensor` of shape `(total_unpadded_tokens,)`, *optional*):
            Indices of the non-padding tokens in the input sequence. Used for unpadding the output.
        cu_seqlens (`torch.Tensor` of shape `(batch + 1,)`, *optional*):
            Cumulative sequence lengths of the input sequences. Used to index the unpadded tensors.
        max_seqlen (`int`, *optional*):
            Maximum sequence length in the batch excluding padding tokens. Used to unpad input_ids & pad output tensors.
        batch_size (`int`, *optional*):
            Batch size of the input sequences. Used to pad the output tensors.
        seq_len (`int`, *optional*):
            Sequence length of the input sequences including padding tokens. Used to pad the output tensors.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        self._maybe_set_compile()

        if input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)

        if batch_size is None and seq_len is None:
            if inputs_embeds is not None:
                batch_size, seq_len = inputs_embeds.shape[:2]
            else:
                batch_size, seq_len = input_ids.shape[:2]  # type: ignore
        device = input_ids.device if input_ids is not None else inputs_embeds.device  # type: ignore

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), device=device, dtype=torch.bool)  # type: ignore

        # Ensure output_hidden_states is True in case pooling mode requires all hidden states
        output_hidden_states = True

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            indices=indices,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            batch_size=batch_size,
            seq_len=seq_len,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = outputs[0]
        hidden_states = outputs[1]

        last_hidden_state = _pool_modchembert_output(
            self,
            last_hidden_state,
            hidden_states,
            typing.cast(torch.Tensor, attention_mask),
        )
        pooled_output = self.head(last_hidden_state)
        pooled_output = self.drop(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def _pool_modchembert_output(
    module: ModChemBertForSequenceClassification,
    last_hidden_state: torch.Tensor,
    hidden_states: list[torch.Tensor],
    attention_mask: torch.Tensor,
):
    """
    Apply pooling strategy to hidden states for sequence-level classification/regression tasks.

    This function implements various pooling strategies to aggregate sequence representations
    into a single vector for downstream classification or regression tasks. The pooling method
    is determined by the `classifier_pooling` configuration parameter.

    Available pooling strategies:
    - cls: Use the CLS token ([CLS]) representation from the last hidden state
    - mean: Average pooling over all tokens in the sequence (attention-weighted)
    - max_cls: Element-wise max pooling over the last k hidden states, then take CLS token
    - cls_mha: Multi-head attention with CLS token as query and full sequence as keys/values
    - max_seq_mha: Max pooling over last k states + multi-head attention with CLS as query
    - max_seq_mean: Max pooling over last k hidden states, then mean pooling over sequence
    - sum_mean: Sum all hidden states across layers, then mean pool over sequence
    - sum_sum: Sum all hidden states across layers, then sum pool over sequence
    - mean_sum: Mean all hidden states across layers, then sum pool over sequence
    - mean_mean: Mean all hidden states across layers, then mean pool over sequence

    Args:
        module: The model instance containing configuration and pooling attention if needed
        last_hidden_state: Final layer hidden states of shape (batch_size, seq_len, hidden_size)
        hidden_states: List of hidden states from all layers, each of shape (batch_size, seq_len, hidden_size)
        attention_mask: Attention mask of shape (batch_size, seq_len) indicating valid tokens

    Returns:
        torch.Tensor: Pooled representation of shape (batch_size, hidden_size)

    Note:
        Some pooling strategies (cls_mha, max_seq_mha) require the module to have a pooling_attn
        attribute containing a ModChemBertPoolingAttention instance.
    """
    config = typing.cast(ModChemBertConfig, module.config)
    if config.classifier_pooling == "cls":
        last_hidden_state = last_hidden_state[:, 0]
    elif config.classifier_pooling == "mean":
        last_hidden_state = (last_hidden_state * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(
            dim=1, keepdim=True
        )
    elif config.classifier_pooling == "max_cls":
        k_hidden_states = hidden_states[-config.classifier_pooling_last_k :]
        theta = torch.stack(k_hidden_states, dim=1)  # (batch, k, seq_len, hidden)
        pooled_seq = torch.max(theta, dim=1).values  # Element-wise max over k -> (batch, seq_len, hidden)
        last_hidden_state = pooled_seq[:, 0, :]  # (batch, hidden)
    elif config.classifier_pooling == "cls_mha":
        # Similar to max_seq_mha but without the max pooling step
        # Query is CLS token (position 0); Keys/Values are full sequence
        q = last_hidden_state[:, 0, :].unsqueeze(1)  # (batch, 1, hidden)
        q = q.expand(-1, last_hidden_state.shape[1], -1)  # (batch, seq_len, hidden)
        attn_out: torch.Tensor = module.pooling_attn(  # type: ignore
            q=q, kv=last_hidden_state, attention_mask=attention_mask
        )  # (batch, seq_len, hidden)
        last_hidden_state = torch.mean(attn_out, dim=1)
    elif config.classifier_pooling == "max_seq_mha":
        k_hidden_states = hidden_states[-config.classifier_pooling_last_k :]
        theta = torch.stack(k_hidden_states, dim=1)  # (batch, k, seq_len, hidden)
        pooled_seq = torch.max(theta, dim=1).values  # Element-wise max over k -> (batch, seq_len, hidden)
        # Query is pooled CLS token (position 0); Keys/Values are pooled sequence
        q = pooled_seq[:, 0, :].unsqueeze(1)  # (batch, 1, hidden)
        q = q.expand(-1, pooled_seq.shape[1], -1)  # (batch, seq_len, hidden)
        attn_out: torch.Tensor = module.pooling_attn(  # type: ignore
            q=q, kv=pooled_seq, attention_mask=attention_mask
        )  # (batch, seq_len, hidden)
        last_hidden_state = torch.mean(attn_out, dim=1)
    elif config.classifier_pooling == "max_seq_mean":
        k_hidden_states = hidden_states[-config.classifier_pooling_last_k :]
        theta = torch.stack(k_hidden_states, dim=1)  # (batch, k, seq_len, hidden)
        pooled_seq = torch.max(theta, dim=1).values  # Element-wise max over k -> (batch, seq_len, hidden)
        last_hidden_state = torch.mean(pooled_seq, dim=1)  # Mean over sequence length
    elif config.classifier_pooling == "sum_mean":
        # ChemLM uses the mean of all hidden states
        # which outperforms using just the last layer mean or the cls embedding
        # https://doi.org/10.1038/s42004-025-01484-4
        # https://static-content.springer.com/esm/art%3A10.1038%2Fs42004-025-01484-4/MediaObjects/42004_2025_1484_MOESM2_ESM.pdf
        all_hidden_states = torch.stack(hidden_states)
        w = torch.sum(all_hidden_states, dim=0)
        last_hidden_state = torch.mean(w, dim=1)
    elif config.classifier_pooling == "sum_sum":
        all_hidden_states = torch.stack(hidden_states)
        w = torch.sum(all_hidden_states, dim=0)
        last_hidden_state = torch.sum(w, dim=1)
    elif config.classifier_pooling == "mean_sum":
        all_hidden_states = torch.stack(hidden_states)
        w = torch.mean(all_hidden_states, dim=0)
        last_hidden_state = torch.sum(w, dim=1)
    elif config.classifier_pooling == "mean_mean":
        all_hidden_states = torch.stack(hidden_states)
        w = torch.mean(all_hidden_states, dim=0)
        last_hidden_state = torch.mean(w, dim=1)
    return last_hidden_state


__all__ = [
    "ModChemBertForMaskedLM",
    "ModChemBertForSequenceClassification",
]
