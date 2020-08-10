
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.modules import LayerNorm, MultiheadAttention
from fairseq.modules.group_linear_layer import GroupLinearLayer

from fairseq.modules.quant_noise import quant_noise
from fairseq.modules.fairseq_dropout import FairseqDropout
from torch import Tensor

import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, n_heads, n_blocks, dim):
        super(Attention, self).__init__()

        #self.n_heads = n_heads
        self.n_heads = 2
        self.n_blocks = n_blocks
        self.dim = dim
        self.block_dim = dim // self.n_blocks
        #self.head_dim = self.block_dim // self.n_heads
        self.head_dim = 32
        self.scale = self.head_dim ** -0.5

        self.query_net = nn.Linear(self.block_dim, self.head_dim * self.n_heads)
        self.key_net = nn.Linear(self.block_dim, self.head_dim * self.n_heads)
        self.value_net = nn.Linear(self.block_dim, self.head_dim * self.n_heads)
        self.final = nn.Linear(self.head_dim * self.n_heads, self.block_dim)

    def forward(self, x):
        seq_len, bsz, _ = x.shape
        x = x.view(seq_len, bsz, self.n_blocks, self.block_dim)

        q = self.query_net(x).view(seq_len, bsz, self.n_blocks, self.n_heads, self.head_dim)
        k = self.key_net(x).view(seq_len, bsz, self.n_blocks, self.n_heads, self.head_dim)
        v = self.value_net(x).view(seq_len, bsz, self.n_blocks, self.n_heads, self.head_dim)
        
        q = q.transpose(2,3) * self.scale
        k = k.transpose(2,3)
        v = v.transpose(2,3)

        score = torch.matmul(q, k.transpose(3,4))
        score = F.softmax(score, dim=-1)
        out = torch.matmul(score, v).transpose(2,3)
        score = score.mean(dim=2)

        out = out.reshape(seq_len, bsz, self.n_blocks, self.head_dim * self.n_heads)
        out = self.final(out)
        out = out.view(seq_len, bsz, self.dim)

        return out, score 

class NormLayer(nn.Module):
    def __init__(self, num_rims, dim, export=False):
        super(NormLayer, self).__init__()

        self.num_rims = num_rims
        self.dim = dim

        self.norm = LayerNorm(dim, export=export)

    def forward(self, x):
        seq_len, bsz, _ = x.shape
        x = x.view(seq_len, bsz, self.num_rims, self.dim)

        x = self.norm(x)
        
        x = x.view(seq_len, bsz, self.num_rims * self.dim)

        return x
 
class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args):
        super().__init__()
        self.blockatt = True
        self.embed_dim = args.encoder_embed_dim
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)
 
        print('encoder embed_dim', self.embed_dim)

        self.nb = args.num_modules
        self.norm_blocks = args.num_modules

        self.self_attn = self.build_self_attention(self.embed_dim, args) #should divide embed_dim by nb.  Then raise embed_dim in args
        self.self_attn_layer_norm = NormLayer(self.norm_blocks, self.embed_dim // self.norm_blocks)
        self.dropout_module = FairseqDropout(args.dropout, module_name=self.__class__.__name__)
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, "activation_fn", "relu")
        )

        print("SETUP TRANSFORMER LAYER", 'blocks', self.nb)

        activation_dropout_p = getattr(args, "activation_dropout", 0)
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0)
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = self.build_fc1(
            self.embed_dim, args.encoder_ffn_embed_dim, self.quant_noise, self.quant_noise_block_size
        )
        self.fc2 = self.build_fc2(
            args.encoder_ffn_embed_dim, self.embed_dim, self.quant_noise, self.quant_noise_block_size
        )

        self.final_layer_norm = NormLayer(self.norm_blocks, self.embed_dim // self.norm_blocks)

        if self.blockatt:
            self.comm = Attention(args.encoder_attention_heads, self.nb, self.embed_dim)
            self.comm_norm = NormLayer(self.norm_blocks, self.embed_dim // self.norm_blocks)

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(GroupLinearLayer(input_dim//self.nb, output_dim//self.nb, self.nb), p=q_noise, block_size=qn_block_size)
        #return quant_noise(nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(GroupLinearLayer(input_dim//self.nb, output_dim//self.nb, self.nb), p=q_noise, block_size=qn_block_size)
        #return quant_noise(nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size)

    def build_self_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim//self.nb,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            nblocks=self.nb,
            top_k = int(args.topk_ratio * args.encoder_attention_heads * self.nb)
        )

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]

    def forward(self, x, encoder_padding_mask, attn_mask: Optional[Tensor] = None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters

        seq_len, bsz, _ = x.shape

        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        
        x, _ = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=encoder_padding_mask,
                attn_mask=attn_mask,
            )

        x = self.dropout_module(x)
        x = residual + x

        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.blockatt:
            if self.normalize_before:
                x = self.comm_norm(x)
            
            residual = x
            x, _ = self.comm(x)
            x = self.dropout_module(x)
            x = residual + x
            
            if not self.normalize_before:
                x = self.comm_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        #print('fc1 on shape', x.shape, 'in encoder')
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        #print('fc2 on shape', x.shape, 'in encoder')
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x


class TransformerDecoderLayer(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__()
        self.blockatt = True
        self.embed_dim = args.decoder_embed_dim
        self.dropout_module = FairseqDropout(args.dropout, module_name=self.__class__.__name__)
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)

        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        self.nb = args.num_modules
        self.norm_blocks = args.num_modules
        print("SETUP TRANSFORMER DECODER LAYER")

        self.self_attn = self.build_self_attention(
            self.embed_dim,
            args,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )
        if self.blockatt:
            self.self_comm = Attention(args.decoder_attention_heads, self.nb, self.embed_dim)
            self.self_comm_norm = NormLayer(self.norm_blocks, self.embed_dim // self.norm_blocks)

        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, "activation_fn", "relu")
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0)
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0)
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__)
        self.normalize_before = args.decoder_normalize_before

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        export = getattr(args, "char_inputs", False)
        self.self_attn_layer_norm = NormLayer(self.norm_blocks, self.embed_dim // self.norm_blocks, export=export)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
            self.encoder_comm_norm = None
            self.encoder_comm = None
        else:
            self.encoder_attn = self.build_encoder_attention(self.embed_dim, args)
            self.encoder_attn_layer_norm = NormLayer(self.norm_blocks, self.embed_dim // self.norm_blocks, export=export)
            if self.blockatt:
                self.encoder_comm_norm = NormLayer(self.norm_blocks, self.embed_dim // self.norm_blocks)
                self.encoder_comm = Attention(args.decoder_attention_heads, self.nb, self.embed_dim)

        print('setup transformer layer decoder blocks: ', self.nb)

        self.fc1 = self.build_fc1(
            self.embed_dim, args.decoder_ffn_embed_dim, self.quant_noise, self.quant_noise_block_size
        )
        self.fc2 = self.build_fc2(
            args.decoder_ffn_embed_dim, self.embed_dim, self.quant_noise, self.quant_noise_block_size
        )

        self.final_layer_norm = NormLayer(self.norm_blocks, self.embed_dim // self.norm_blocks, export=export)
        self.need_attn = True

        self.onnx_trace = False

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(GroupLinearLayer(input_dim//self.nb, output_dim//self.nb, self.nb), q_noise, qn_block_size)
        #return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(GroupLinearLayer(input_dim//self.nb, output_dim//self.nb, self.nb), q_noise, qn_block_size)
        #return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_self_attention(self, embed_dim, args, add_bias_kv=False, add_zero_attn=False):
        return MultiheadAttention(
            embed_dim//self.nb,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not getattr(args, "cross_self_attention", False),
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            nblocks=self.nb,
            top_k = int(args.topk_ratio * args.decoder_attention_heads * self.nb)
        )

    def build_encoder_attention(self, embed_dim, args):
        kdim = getattr(args, "encoder_embed_dim", None)
        vdim = getattr(args, "encoder_embed_dim", None)

        if kdim is not None:
            kdim = kdim//self.nb
        if vdim is not None:
            vdim = vdim//self.nb

        return MultiheadAttention(
            embed_dim//self.nb,
            args.decoder_attention_heads,
            kdim=kdim,
            vdim=vdim,
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            nblocks=self.nb,
            top_k = int(args.topk_ratio * args.decoder_attention_heads * self.nb)
        )

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
            incremental_state is not None
            and _self_attn_input_buffer is not None
            and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )

        x = self.dropout_module(x)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.blockatt:
            if self.normalize_before:
                x = self.self_comm_norm(x)

            residual = x
            x, _ = self.self_comm(x)
            x = self.dropout_module(x)
            x = residual + x
            
            if not self.normalize_before:
                x = self.self_comm_norm(x)

        if self.encoder_attn is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)
            
            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )

            x = self.dropout_module(x)
            x = residual + x
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

            if self.blockatt:
                if self.normalize_before:
                    x = self.encoder_comm_norm(x)

                residual = x
                x, _ = self.encoder_comm(x)
                x = self.dropout_module(x)
                x = residual + x
                
                if not self.normalize_before:
                    x = self.encoder_comm_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None

    def make_generation_fast_(self, need_attn: bool = False, **kwargs):
        self.need_attn = need_attn


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


