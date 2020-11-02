
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

from fairseq import utils
from group_linear_layer import GroupLinearLayer

#note: need to change init_bert_params to handle self.in_proj_q, self.in_proj_k, self.in_proj_v

class MultiheadAttention(nn.Module):
    """MultiHeadAttention
    """

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, scaling_factor=1, nblocks=1, top_k_ratio=None):
        super().__init__()
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        self.nblocks = nblocks
        self.top_k_ratio = top_k_ratio

        #self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, self.head_dim * num_heads))

        self.in_proj_q = GroupLinearLayer(embed_dim//self.nblocks, self.head_dim * num_heads // self.nblocks, self.nblocks)
        self.in_proj_k = GroupLinearLayer(embed_dim//self.nblocks, self.head_dim * num_heads // self.nblocks, self.nblocks)
        self.in_proj_v = GroupLinearLayer(embed_dim//self.nblocks, self.head_dim * num_heads // self.nblocks, self.nblocks)

        #if bias:
        #    self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        #else:
        #    self.register_parameter('in_proj_bias', None)

        self.out_proj = GroupLinearLayer(self.head_dim * num_heads // self.nblocks, embed_dim // self.nblocks, self.nblocks)

        self.scaling_factor = scaling_factor

        self.reset_parameters()


    def reset_parameters(self):
        # Note: these initilaztion will be overrided in `init_bert_params`, if using BERT
        #nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        #if self.in_proj_bias is not None:
        #    nn.init.constant_(self.in_proj_bias, 0.)

    def forward(
        self,
        query, key, value,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        attn_bias=None
    ):
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
        """
        tgt_len, bsz, embed_dim = query.size()
        src_len = tgt_len
        embed_dim_per_module = embed_dim // self.nblocks

        #query = query.reshape(tgt_len, bsz * self.nblocks, embed_dim_per_module)

        #if key is not None:
        #    src_len = key.shape[0]
        #    key = key.reshape(src_len, bsz * self.nblocks, embed_dim_per_module)
        #    value = value.reshape(src_len, bsz * self.nblocks, embed_dim_per_module)

        kp_mask = None
        if key_padding_mask is not None:
            b, s = key_padding_mask.shape
            kp_mask = key_padding_mask#.view(b, 1, s).rep.view(b * self.nblocks, s)

        head_dim = embed_dim // self.num_heads
        scaling = float(head_dim * self.scaling_factor) ** -0.5

        #q, k, v = F.linear(query, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        
        q = self.in_proj_q(query)
        k = self.in_proj_k(query)
        v = self.in_proj_v(query)
        
        q = q * scaling

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, head_dim).transpose(0, 1)
        print('k shape', k.shape, 'bsz/src_len,heads,head_dim', bsz, src_len, self.num_heads, head_dim)
        if k is not None:
            k = k.contiguous().view(src_len, bsz * self.num_heads, head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(src_len, bsz * self.num_heads, head_dim).transpose(0, 1)

        src_len = k.size(1)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        attn_output_weights = torch.bmm(q, k.transpose(1, 2))

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_output_weights.masked_fill_(attn_mask, float('-inf'))
            else:
                attn_output_weights += attn_mask

        if attn_bias is not None:
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_output_weights += attn_bias
            attn_output_weights = attn_output_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            )
            attn_output_weights = attn_output_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_output_weights = F.softmax(attn_output_weights, dim=-1, dtype=torch.float32).type_as(query)
        attn_output_weights = F.dropout(attn_output_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_output_weights, v)

        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, head_dim * self.num_heads)
        print(attn_output.shape)
        attn_output = self.out_proj(attn_output)

        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
            return attn_output, attn_output_weights.sum(dim=1) / self.num_heads
        else:
            return attn_output, None

if __name__ == "__main__":

    mha = MultiheadAttention(embed_dim=512, num_heads=4, nblocks=2)

    b,Tq,m = 20,10,512
    Ts = 10

    q = torch.randn(Tq,b,m)
    k = torch.randn(Ts,b,m)
    v = torch.randn(Ts,b,m)

    out,_ = mha(q,k,v)

    print('out shape', out.shape)


