import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.distributed import fsdp_wrap
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    AdaptiveSoftmax,
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from fairseq.models.transformer import TransformerDecoder
from .TransformerBackwardDecoderLayer import TransformerBackwardDecoderLayer
from torch import Tensor


DEFAULT_MIN_PARAMS_TO_WRAP = int(1e8)

class TransformerBackwardDecoder(TransformerDecoder):

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        self.args = args
        self._future_backward_mask = torch.empty(0)
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn=False)

    def build_decoder_layer(self, args, no_encoder_attn=False):
        layer = TransformerBackwardDecoderLayer(args, no_encoder_attn)
        checkpoint = getattr(args, "checkpoint_activations", False)
        if checkpoint:
            offload_to_cpu = getattr(args, "offload_activations", False)
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = (
            getattr(args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP)
            if not checkpoint else 0
        )
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    def forward(
        self,
        back_prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
    ):
        x, extra = self.extract_features(
            back_prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            return_all_hiddens=return_all_hiddens,
        )
        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def extract_features(
        self,
        back_prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        return_all_hiddens: bool = False,
    ):
        return self.extract_features_scriptable(
            back_prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
            return_all_hiddens,
        )


    def extract_features_scriptable(
        self,
        back_prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        return_all_hiddens: bool = False,
    ):
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        # embed positions
        back_positions = None
        if self.embed_positions is not None:
            back_positions = self.embed_positions(
                back_prev_output_tokens, incremental_state=incremental_state
            )

        if incremental_state is not None:
            back_prev_output_tokens = back_prev_output_tokens[:, -1:]
            if back_positions is not None:
                back_positions = back_positions[:, -1:]

        # embed tokens and positions
        backward_x = self.embed_scale * self.embed_tokens(back_prev_output_tokens)

        if self.quant_noise is not None:
            backward_x = self.quant_noise(backward_x)

        if self.project_in_dim is not None:
            backward_x = self.project_in_dim(backward_x)

        if back_positions is not None:
            backward_x += back_positions

        if self.layernorm_embedding is not None:
            backward_x = self.layernorm_embedding(backward_x)

        backward_x = self.dropout_module(backward_x)

        # B x T x C -> T x B x C
        backward_x = backward_x.transpose(0, 1)

        back_self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or back_prev_output_tokens.eq(self.padding_idx).any():
            back_self_attn_padding_mask = back_prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = []
        if return_all_hiddens:
            inner_states.append(backward_x)
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_backward_mask = self.buffered_backward_future_mask(backward_x)
            else:
                self_attn_backward_mask = None

            backward_x, layer_attn, _ = layer(
                    backward_x,
                    encoder_out["encoder_out"][0]
                    if (encoder_out is not None and len(encoder_out["encoder_out"]) > 0)
                    else None,
                    encoder_out["encoder_padding_mask"][0]
                    if (
                        encoder_out is not None
                        and len(encoder_out["encoder_padding_mask"]) > 0
                    )
                    else None,
                    incremental_state,
                    self_attn_mask=self_attn_backward_mask,
                    self_attn_padding_mask=back_self_attn_padding_mask,
                    need_attn=bool((idx == alignment_layer)),
                    need_head_weights=bool((idx == alignment_layer)),
            )
            if return_all_hiddens:
                assert inner_states is not None
                inner_states.append(backward_x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(backward_x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            backward_x = self.layer_norm(backward_x)

        # T x B x C -> B x T x C
        backward_x = backward_x.transpose(0, 1)

        if self.project_out_dim is not None:
            backward_x = self.project_out_dim(backward_x)

        return backward_x, {"attn": [attn], "inner_states": inner_states}

    def output_layer(self, features):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            return self.output_projection(features)
        else:
            return features

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)  # T
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
            self._future_mask.size(0) == 0
            or (not self._future_mask.device == tensor.device)
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(torch.zeros([dim, dim])), 1
            )
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]

    def buffered_backward_future_mask(self, tensor):
        '''Return a lower triangular matrix'''
        dim = tensor.size(0)  # T
        # self._future_backward_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
            self._future_backward_mask.size(0) == 0
            or (not self._future_backward_mask.device == tensor.device)
            or self._future_backward_mask.size(0) < dim
        ):
            self._future_backward_mask = torch.tril(
                utils.fill_with_neg_inf(torch.zeros([dim, dim])), -1
            )
        self._future_backward_mask = self._future_backward_mask.to(tensor)
        return self._future_backward_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)

        if f"{name}.output_projection.weight" not in state_dict:
            if self.share_input_output_embed:
                embed_out_key = f"{name}.embed_tokens.weight"
            else:
                embed_out_key = f"{name}.embed_out"
            if embed_out_key in state_dict:
                state_dict[f"{name}.output_projection.weight"] = state_dict[
                    embed_out_key
                ]
                if not self.share_input_output_embed:
                    del state_dict[embed_out_key]

        for i in range(self.num_layers):
            # update layer norms
            layer_norm_map = {
                "0": "self_attn_layer_norm",
                "1": "encoder_attn_layer_norm",
                "2": "final_layer_norm",
            }
            for old, new in layer_norm_map.items():
                for m in ("weight", "bias"):
                    k = "{}.layers.{}.layer_norms.{}.{}".format(name, i, old, m)
                    if k in state_dict:
                        state_dict[
                            "{}.layers.{}.{}.{}".format(name, i, new, m)
                        ] = state_dict[k]
                        del state_dict[k]

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) <= 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])

        return state_dict


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m

