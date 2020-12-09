


class AttentiveNFM(nn.Module):
    def __init__(self, n_heads, n_blocks, dim):
        super(AttentiveNFM, self).__init__()

        self.norm = NormLayer(n_blocks, dim // n_blocks)

        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.dim = dim
        self.block_dim = dim // self.n_blocks
        #self.head_dim = self.block_dim // self.n_heads
        self.head_dim = 64
        self.scale = self.head_dim ** -0.5

        self.query_net = GroupLinearLayer(self.block_dim, self.head_dim * self.n_heads, n_blocks)
        self.key_net = GroupLinearLayer(self.block_dim, self.head_dim * self.n_heads, n_blocks)
        self.value_net = GroupLinearLayer(self.block_dim, self.head_dim * self.n_heads, n_blocks)

        self.final = GroupLinearLayer(self.head_dim * self.n_heads, self.block_dim, n_blocks)

    def reset(self):
        self.klst = []
        self.vlst = []

    '''
        Send in a result from a function to update klst and vlst.  
    '''
    def update_kv(self, x): 
        seq_len, bsz, _ = x.shape
        k = self.key_net(x).view(seq_len, bsz, self.n_blocks, self.n_heads, self.head_dim)
        v = self.value_net(x).view(seq_len, bsz, self.n_blocks, self.n_heads, self.head_dim)
        k = k.transpose(2,3)
        v = v.transpose(2,3)
        self.klst.append(k)
        self.vlst.append(v)


    '''
        Update the current layer by attending over the saved self.klst and self.vlst
    '''
    def forward(self, x):
        residual = x*1.0

        x = self.norm(x)

        klst = self.klst
        vlst = self.vlst

        seq_len, bsz, _ = x.shape

        x = x.view(seq_len, bsz, self.n_blocks * self.block_dim)
        q = self.query_net(x).view(seq_len, bsz, self.n_blocks, self.n_heads, self.head_dim)

        q = q.transpose(2,3) * self.scale

        k = torch.cat(klst, dim=3)
        v = torch.cat(vlst, dim=3)

        score = torch.matmul(q, k.transpose(3,4))
        score = F.softmax(score, dim=-1)
        out = torch.matmul(score, v).transpose(2,3)
        score = score.mean(dim=2)

        out = out.reshape(seq_len, bsz, self.n_blocks * self.head_dim * self.n_heads)
        out = self.final(out)
        out = out.view(seq_len, bsz, self.dim)

        out = residual + out

        return out



        #self.self_mem_norm = NormLayer(self.norm_blocks, self.embed_dim // self.norm_blocks)

        #self.memory_layer = RelationalMemory(mem_slots=5, head_size=32, input_size=self.embed_dim, output_size=self.embed_dim, num_heads=4, num_blocks=1, forget_bias=1., input_bias=0., gate_style='unit')

        #self.memory_attention = MemoryAttention(n_blocks_query=self.nb, n_blocks_val=10, dim_query=self.embed_dim, dim_val=5*32*4)

#        if self.blockatt:
#            if self.normalize_before:
#                x = self.self_mem_norm(x)
#            residual = x
#            T,bsz,nhid = x.shape
#            if comp is not None:
#                x_write = comp * x
#            else:
#                x_write = x*1.0
#            _, new_memory = self.memory_layer.forward_step(x_write.reshape((T*bsz, nhid)), self.memory_obj[0])
#            self.memory_obj[0] = new_memory
#            Tbs,num_slots,nhid_slot = new_memory.shape
#            mem_read = new_memory.reshape((T, bsz, num_slots*nhid_slot))
#            x,_ = self.memory_attention(x, mem_read)
#            x = residual + x

#            if not self.normalize_before:
#                x = self.self_mem_norm(x)



