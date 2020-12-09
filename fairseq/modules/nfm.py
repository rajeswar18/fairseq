


class AttentiveNFM(nn.Module):
    def __init__(self, n_heads, n_blocks, dim, use_nfm):
        super(Attention, self).__init__()

        self.use_nfm = use_nfm

        #self.n_heads = n_heads
        self.n_heads = 12
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

