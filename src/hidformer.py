import torch
import torch.nn as nn
import torch.nn.functional as F


class RevIN(nn.Module):
    """
    Reversible Instance Normalization (Shen et al., 2023).
    Normalizes per-instance, per-channel statistics, then denormalizes output.
    """
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(1, num_features, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_features, 1))

    def forward(self, x, mode='norm'):
        # x: (B, C, T)
        if mode == 'norm':
            self.mean = x.mean(dim=-1, keepdim=True)
            self.std = x.std(dim=-1, keepdim=True)
            x_norm = (x - self.mean) / (self.std + self.eps)
            if self.affine:
                x_norm = x_norm * self.weight + self.bias
            return x_norm
        elif mode == 'denorm':
            x_denorm = (x - (self.bias if self.affine else 0)) / (self.weight if self.affine else 1)
            return x_denorm * (self.std + self.eps) + self.mean
        else:
            raise ValueError("mode must be 'norm' or 'denorm'")


class SegmentMerge(nn.Module):
    """
    Performs initial segmentation into tokens and hierarchical merging across multiple scales.
    """
    def __init__(self, token_length: int, stride: int, num_blocks: int):
        super().__init__()
        self.token_length = token_length
        self.stride = stride
        self.num_blocks = num_blocks

    def forward(self, x):  # x: (B, C, T)
        B, C, T = x.shape
        # 1) segmentation -> (B, num_tokens, C, token_length)
        segments = x.unfold(-1, self.token_length, self.stride)  # (B, C, N, token_length)
        segments = segments.permute(0, 2, 1, 3).contiguous()       # (B, N, C, token_length)
        tokens = segments.view(B, segments.size(1), -1)             # (B, N, C*token_length)

        # 2) hierarchical merging across blocks
        merged = tokens
        out_tokens = []
        for _ in range(self.num_blocks):
            out_tokens.append(merged)
            N = merged.size(1)
            if N % 2 == 1:
                merged = torch.cat([merged, merged[:, -1:].clone()], dim=1)
                N += 1
            # merge adjacent pairs: average
            merged = merged.view(B, N // 2, -1)
        return out_tokens  # list of length num_blocks, each (B, Ni, token_dim)


class SRUPlusPlus(nn.Module):
    """
    SRU++: combines self-attention-based U projection and parallel recurrence.
    """
    def __init__(self, d_model, hidden_size):
        super().__init__()
        # weights for gates
        self.W = nn.Parameter(torch.Tensor(3 * hidden_size, d_model))
        self.vf = nn.Parameter(torch.Tensor(hidden_size))
        self.vr = nn.Parameter(torch.Tensor(hidden_size))
        self.bf = nn.Parameter(torch.Tensor(hidden_size))
        self.br = nn.Parameter(torch.Tensor(hidden_size))
        # self-attention for U
        self.to_q = nn.Linear(d_model, hidden_size, bias=False)
        self.to_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wu = nn.Linear(hidden_size, 3 * hidden_size)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.hidden_size = hidden_size
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.zeros_(self.vf)
        nn.init.zeros_(self.vr)
        nn.init.zeros_(self.bf)
        nn.init.zeros_(self.br)

    def forward(self, x):  # x: (B, N, d_model)
        B, N, _ = x.size()
        # compute U via self-attention mechanism
        Z = x.transpose(1, 2)                 # (B, d_model, N)
        Q = self.to_q(x)                      # (B, N, hidden)
        K = self.to_k(Q)                      # (B, N, hidden)
        V = self.to_v(Q)                      # (B, N, hidden)
        attn = torch.softmax((Q @ K.transpose(-1, -2)) / self.hidden_size**0.5, dim=-1)
        F = attn @ V                          # (B, N, hidden)
        U = self.Wu(Q + self.alpha * F)       # (B, N, 3*hidden)
        U = U.transpose(1, 2)                 # (B, 3*hidden, N)

        # parallel recurrence across hidden dims
        U0, U1, U2 = U.chunk(3, dim=1)        # each (B, hidden, N)
        h = U2                                 # candidate
        c_prev = torch.zeros(B, self.hidden_size, device=x.device)
        c_list = []
        for t in range(N):
            u0_t = U0[:, :, t]
            u1_t = U1[:, :, t]
            u2_t = U2[:, :, t]
            f_t = torch.sigmoid(u0_t + self.vf * c_prev + self.bf)
            c_t = f_t * c_prev + (1 - f_t) * u2_t
            r_t = torch.sigmoid(u1_t + self.vr * c_prev + self.br)
            h_t = r_t * c_t + (1 - r_t) * x[:, t].transpose(1,0)
            c_prev = c_t
            c_list.append(h_t)
        H = torch.stack(c_list, dim=1)         # (B, N, hidden)
        # project back to model dimension
        out = H @ self.W                       # (B, N, d_model)
        return out


class LinearAttention(nn.Module):
    """
    Linear self-attention (Wang et al., 2020) with low-rank projections.
    """
    def __init__(self, d_model, k):
        super().__init__()
        self.to_q = nn.Linear(d_model, d_model, bias=False)
        self.to_k = nn.Linear(d_model, d_model, bias=False)
        self.to_v = nn.Linear(d_model, d_model, bias=False)
        self.A = nn.Parameter(torch.randn(d_model, k))
        # share A for B
        self.B = self.A

    def forward(self, x):  # x: (B, N, d_model)
        Q = self.to_q(x)  # (B, N, d)
        K = self.to_k(x)
        V = self.to_v(x)
        K_hat = K @ self.A  # (B, N, k)
        V_hat = V @ self.B  # (B, N, k)
        scores = torch.softmax((Q @ K_hat.transpose(-1,-2)) / x.size(-1)**0.5, dim=-1)
        out = scores @ V_hat  # (B, N, k)
        # project back
        return nn.Linear(V_hat.size(-1), x.size(-1))(out)


class TimeBlock(nn.Module):
    def __init__(self, d_model, hidden_size):
        super().__init__()
        self.mixer = SRUPlusPlus(d_model, hidden_size)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, N, d_model)
        y = self.mixer(x)
        x = x + y
        x = self.norm1(x)
        y2 = self.ffn(x)
        x = x + y2
        return self.norm2(x)


class FrequencyBlock(nn.Module):
    def __init__(self, d_model, k):
        super().__init__()
        self.mixer = LinearAttention(d_model, k)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        y = self.mixer(x)
        x = x + y
        x = self.norm1(x)
        y2 = self.ffn(x)
        x = x + y2
        return self.norm2(x)


class Hidformer(nn.Module):
    def __init__(self,
                 input_dim,
                 token_length=16,
                 stride=8,
                 time_blocks=4,
                 freq_blocks=2,
                 hidden_size=128,
                 freq_k=64,
                 dout=0.2,
                 out_dim=None):
        super().__init__()
        self.rev_in = RevIN(input_dim)
        self.segmenter = SegmentMerge(token_length, stride, max(time_blocks, freq_blocks))
        token_dim = input_dim * token_length
        self.time_tower = nn.ModuleList([
            TimeBlock(token_dim, hidden_size) for _ in range(time_blocks)
        ])
        self.freq_tower = nn.ModuleList([
            FrequencyBlock(token_dim, freq_k) for _ in range(freq_blocks)
        ])
        # adaptors
        self.time_adapt = nn.Linear(token_dim, token_dim)
        self.freq_adapt = nn.Linear(token_dim, token_dim)
        # final decoder
        total_feats = token_dim * (time_blocks + freq_blocks)
        self.decoder = nn.Linear(total_feats, out_dim or input_dim)

    def forward(self, x):
        # x: (B, T, C) -> (B, C, T)
        x = x.transpose(1, 2)
        x_norm = self.rev_in(x, mode='norm')
        # segmentation + hierarchical outputs
        multi_tokens = self.segmenter(x_norm)  # list of length L: (B, Ni, token_dim)
        t_feats = []
        for tokens, block in zip(multi_tokens[:len(self.time_tower)], self.time_tower):
            out = block(tokens)  # (B, Ni, dim)
            t_feats.append(self.time_adapt(out.mean(dim=1)))
        # frequency tower: FFT then blocks
        fft_tokens = torch.fft.rfft(multi_tokens[0], dim=1).real
        f_feats = []
        for tokens, block in zip([fft_tokens] + multi_tokens[1:len(self.freq_tower)], self.freq_tower):
            out = block(tokens)
            f_feats.append(self.freq_adapt(out.mean(dim=1)))
        # concat and decode
        all_feats = torch.cat(t_feats + f_feats, dim=-1)
        pred = self.decoder(all_feats)
        # denormalize -> (B, C, T')
        # Here pred is (B, C) for horizon prediction; adapt as needed
        return pred
