# Copyright 2025 Emmanuel Cortes, All Rights Reserved.
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

from typing import Literal

from transformers.models.modernbert.configuration_modernbert import ModernBertConfig


class ModChemBertConfig(ModernBertConfig):
    """
    Configuration class for ModChemBert models.

    This configuration class extends ModernBertConfig with additional parameters specific to
    chemical molecule modeling and custom pooling strategies for classification/regression tasks.
    It accepts all arguments and keyword arguments from ModernBertConfig.

    Args:
        classifier_pooling (str, optional): Pooling strategy for sequence classification.
            Available options:
            - "cls": Use CLS token representation
            - "mean": Attention-weighted average pooling
            - "sum_mean": Sum all hidden states across layers, then mean pool over sequence (ChemLM approach)
            - "sum_sum": Sum all hidden states across layers, then sum pool over sequence
            - "mean_mean": Mean all hidden states across layers, then mean pool over sequence
            - "mean_sum": Mean all hidden states across layers, then sum pool over sequence
            - "max_cls": Element-wise max pooling over last k hidden states, then take CLS token
            - "cls_mha": Multi-head attention with CLS token as query and full sequence as keys/values
            - "max_seq_mha": Max pooling over last k states + multi-head attention with CLS as query
            - "max_seq_mean": Max pooling over last k hidden states, then mean pooling over sequence
            Defaults to "sum_mean".
        classifier_pooling_num_attention_heads (int, optional): Number of attention heads for multi-head attention
            pooling strategies (cls_mha, max_seq_mha). Defaults to 4.
        classifier_pooling_attention_dropout (float, optional): Dropout probability for multi-head attention
            pooling strategies (cls_mha, max_seq_mha). Defaults to 0.0.
        classifier_pooling_last_k (int, optional): Number of last hidden layers to use for max pooling
            strategies (max_cls, max_seq_mha, max_seq_mean). Defaults to 8.
        *args: Variable length argument list passed to ModernBertConfig.
        **kwargs: Arbitrary keyword arguments passed to ModernBertConfig.

    Note:
        This class inherits all configuration parameters from ModernBertConfig including
        hidden_size, num_hidden_layers, num_attention_heads, intermediate_size, etc.
    """

    model_type = "modchembert"

    def __init__(
        self,
        *args,
        classifier_pooling: Literal[
            "cls",
            "mean",
            "sum_mean",
            "sum_sum",
            "mean_mean",
            "mean_sum",
            "max_cls",
            "cls_mha",
            "max_seq_mha",
            "max_seq_mean",
        ] = "max_seq_mha",
        classifier_pooling_num_attention_heads: int = 4,
        classifier_pooling_attention_dropout: float = 0.0,
        classifier_pooling_last_k: int = 8,
        **kwargs,
    ):
        # Pass classifier_pooling="cls" to circumvent ValueError in ModernBertConfig init
        super().__init__(*args, classifier_pooling="cls", **kwargs)
        # Override with custom value
        self.classifier_pooling = classifier_pooling
        self.classifier_pooling_num_attention_heads = classifier_pooling_num_attention_heads
        self.classifier_pooling_attention_dropout = classifier_pooling_attention_dropout
        self.classifier_pooling_last_k = classifier_pooling_last_k
