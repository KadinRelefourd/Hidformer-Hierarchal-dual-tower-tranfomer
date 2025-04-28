import torch
import torch.nn as nn
import torch.nn.functional as F

class RevIN(nn.Module):
    """Reversible Instance Normalization"""
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(1, num_features, 1))
            self.bias   = nn.Parameter(torch.zeros(1, num_features, 1))

    def forward(self, x, mode='norm'):
        # x: (B, C, T)
        if mode == 'norm':
            self.mean = x.mean(dim=-1, keepdim=True)
            self.std  = x.std(dim=-1, keepdim=True)
            x_norm = (x - self.mean) / (self.std + self.eps)
            if self.affine:
                x_norm = x_norm * self.weight + self.bias
            return x_norm
        elif mode == 'denorm':
            x_denorm = (x - (self.bias if self.affine else 0)) / \
                      (self.weight if self.affine else 1)
            return x_denorm * (self.std + self.eps) + self.mean
        else:
            raise ValueError("mode must be 'norm' or 'denorm'")

class SegmentMerge(nn.Module):
    """Hierarchical segmentation & merging."""
    def __init__(self, token_length: int, stride: int, num_blocks: int):
        super().__init__()
        self.token_length = token_length
        self.stride = stride
        self.num_blocks = num_blocks

    def forward(self, x):  # x: (B, C, T)
        B, C, T = x.shape
        # 1) unfold into segments: (B, N, C, token_length)
        seg = x.unfold(-1, self.token_length, self.stride)
        seg = seg.permute(0,2,1,3).contiguous()
        tokens = seg.view(B, seg.size(1), -1)  # (B, N, C*token_length)

        # 2) hierarchical merge
        merged = tokens
        out_list = []
        for _ in range(self.num_blocks):
            out_list.append(merged)
            N = merged.size(1)
            if N % 2 == 1:
                merged = torch.cat([merged, merged[:, -1:].clone()], dim=1)
                N += 1
            merged = merged.view(B, N//2, -1)
        return out_list

class SRUPlusPlus(nn.Module):
    """Self-attention + parallel recurrence"""
    def __init__(self, d_model, hidden_size):
        super().__init__()
        self.W = nn.Parameter(torch.Tensor(3*hidden_size, d_model))
        self.vf = nn.Parameter(torch.Tensor(hidden_size))
        self.vr = nn.Parameter(torch.Tensor(hidden_size))
        self.bf = nn.Parameter(torch.Tensor(hidden_size))
        self.br = nn.Parameter(torch.Tensor(hidden_size))
        self.to_q = nn.Linear(d_model, hidden_size, bias=False)
        self.to_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wu = nn.Linear(hidden_size, 3*hidden_size)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.hidden_size = hidden_size
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.zeros_(self.vf); nn.init.zeros_(self.vr)
        nn.init.zeros_(self.bf); nn.init.zeros_(self.br)

    def forward(self, x):  # x: (B, N, d_model)
        B,N,_ = x.size()
        Q = self.to_q(x)
        K = self.to_k(Q)
        V = self.to_v(Q)
        attn = torch.softmax((Q @ K.transpose(-1,-2)) / (self.hidden_size**0.5), dim=-1)
        Fval = attn @ V
        U = self.Wu(Q + self.alpha * Fval).transpose(1,2)  # (B,3H,N)
        U0, U1, U2 = U.chunk(3, dim=1)

        c_prev = torch.zeros(B, self.hidden_size, device=x.device)
        h_list = []
        for t in range(N):
            u0, u1, u2 = U0[:,:,t], U1[:,:,t], U2[:,:,t]
            f = torch.sigmoid(u0 + self.vf*c_prev + self.bf)
            c = f*c_prev + (1-f)*u2
            r = torch.sigmoid(u1 + self.vr*c_prev + self.br)
            h_t = r*c + (1-r)*x[:,t]
            c_prev = c
            h_list.append(h_t)
        H = torch.stack(h_list, dim=1)
        return H @ self.W

class LinearAttention(nn.Module):
    """Low-rank linear self-attention."""
    def __init__(self, d_model, k):
        super().__init__()
        self.to_q = nn.Linear(d_model, d_model, bias=False)
        self.to_k = nn.Linear(d_model, d_model, bias=False)
        self.to_v = nn.Linear(d_model, d_model, bias=False)
        self.A = nn.Parameter(torch.randn(d_model, k))
        self.B = self.A

    def forward(self, x): # (B,N,d)
        Q = self.to_q(x); K = self.to_k(x); V = self.to_v(x)
        Khat = K @ self.A; Vhat = V @ self.B
        scores = torch.softmax((Q @ Khat.transpose(-1,-2)) / (x.size(-1)**0.5), dim=-1)
        out = scores @ Vhat
        return nn.Linear(Vhat.size(-1), x.size(-1))(out)

class TimeBlock(nn.Module):
    def __init__(self, d_model, hidden_size):
        super().__init__()
        self.mixer = SRUPlusPlus(d_model, hidden_size)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.GELU(),
            nn.Linear(4*d_model, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
    def forward(self, x):
        y = self.mixer(x)
        x = self.norm1(x + y)
        z = self.ffn(x)
        return self.norm2(x + z)

class FrequencyBlock(nn.Module):
    def __init__(self, d_model, k):
        super().__init__()
        self.mixer = LinearAttention(d_model, k)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.GELU(),
            nn.Linear(4*d_model, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
    def forward(self, x):
        y = self.mixer(x)
        x = self.norm1(x + y)
        z = self.ffn(x)
        return self.norm2(x + z)

class Hidformer(nn.Module):
    def __init__(
        self,
        input_dim,
        token_length=16,
        stride=8,
        time_blocks=4,
        freq_blocks=2,
        hidden_size=128,
        freq_k=64,
        dout=0.2,
        out_dim=None
    ):
        super().__init__()
        self.rev_in = RevIN(input_dim)
        self.segmenter = SegmentMerge(token_length, stride, max(time_blocks,freq_blocks))
        token_dim = input_dim * token_length
        self.time_tower = nn.ModuleList([
            TimeBlock(token_dim, hidden_size) for _ in range(time_blocks)
        ])
        self.freq_tower = nn.ModuleList([
            FrequencyBlock(token_dim, freq_k) for _ in range(freq_blocks)
        ])
        self.time_adapt = nn.Linear(token_dim, token_dim)
        self.freq_adapt = nn.Linear(token_dim, token_dim)
        total_feats = token_dim * (time_blocks + freq_blocks)
        self.decoder = nn.Linear(total_feats, out_dim or input_dim)

    def forward(self, x):
        # x: (B, T, C)
        x = x.transpose(1,2)  # (B, C, T)
        xn = self.rev_in(x, mode='norm')
        tokens = self.segmenter(xn)  # list of (B,Ni,token_dim)
        t_feats = []
        for tok, blk in zip(tokens[:len(self.time_tower)], self.time_tower):
            out = blk(tok)
            t_feats.append(self.time_adapt(out.mean(dim=1)))
        fft0 = torch.fft.rfft(tokens[0], dim=1).real
        f_feats = []
        for tok, blk in zip([fft0] + tokens[1:len(self.freq_tower)], self.freq_tower):
            out = blk(tok)
            f_feats.append(self.freq_adapt(out.mean(dim=1)))
        feats = torch.cat(t_feats + f_feats, dim=-1)
        pred = self.decoder(feats)
        return pred
