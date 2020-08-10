
'''
Attn is: (bsz, modules*heads, trg_len, src_len).  

-Idea: compute top-k over all module*heads.  Force any which are below the top-2 value by some margin to go to zero.  

-If there's only 1 step to attend to, then every head will be 100% on that 1-step.  In this case, all should stay on.  

-If 2-steps, could be like 99%, 1%.  Then the 1% gets shut down to zero and re-normalized.  

(modules*heads, bsz, trg_len, src_len)

(bsz*trg_len*src_len

'''


import torch
import torch.nn as nn
import numpy

class SparseAttention(nn.Module):
    def __init__(self, top_k = 5):
        super(SparseAttention,self).__init__()
        #top_k += 1
        print('Sparse attention with topk', top_k)
        self.top_k = top_k

    def forward(self, attn_s):

        bsz, num_modules, trg_len, src_len = attn_s.shape

        time_step = attn_s.size()[3]
        attn_s = attn_s.permute(0,2,3,1).reshape((bsz*trg_len*src_len, num_modules))

        # normalize the attention weights using piece-wise Linear function
        # only top k should
        attn_plot = []
        # torch.max() returns both value and location
        #attn_s_max = torch.max(attn_s, dim = 1)[0]
        #attn_w = torch.clamp(attn_s_max, min = 0, max = attn_s_max)
        eps = 0.0001
        # get top k and return it
        # bottom_k = attn_s.size()[1] - self.top_k
        # value of the top k elements 
        #delta = torch.kthvalue(attn_s, bottm_k, dim= 1 )[0]
        
        delta = torch.topk(attn_s, self.top_k, dim= 1)[0][:,-1]
        #delta = attn_s_max - torch.topk(attn_s, self.top_k, dim= 1)[0][:,-1] + eps
        # normalize
        delta = delta.reshape((delta.shape[0],1))


        attn_w = attn_s * torch.gt(attn_s, delta.repeat(1,num_modules) - eps).float()

        attn_w = attn_w.reshape((bsz, trg_len, src_len, num_modules)).permute(0,3,1,2).reshape((bsz*num_modules*trg_len, src_len)) # bsz, num_modules, trg_len, src_len

        do_normalize = False

        if do_normalize:
            attn_w_sum = torch.sum(attn_w, dim = 1, keepdim=True)
            attn_w_sum = attn_w_sum + eps 
            attn_w_normalize = attn_w / attn_w_sum.repeat(1, time_step)
        else:
            attn_w_normalize = attn_w

        attn_w_normalize = attn_w_normalize.reshape((bsz, num_modules, trg_len, src_len))

        return attn_w_normalize


if __name__ == "__main__":
    k = 1
    print('take top k', k)
    sa = Sparse_attention(top_k=k)

    #batch x time

    #x = torch.from_numpy(numpy.array([[[0.1, 0.0, 0.3, 0.2, 0.4],[0.5,0.4,0.1,0.0,0.0]]]))

    bsz = 1
    num_modules = 4
    trg_len = 1
    src_len = 4

    sm = nn.Softmax(dim=3)
    x = torch.randn((bsz, num_modules, trg_len, src_len))

    x = sm(x)

    #x = x.reshape((2,5))

    print('x shape', x.shape)
    print('x', x)

    o = sa(x)

    print('o shape', o.shape)
    print('o', o)



