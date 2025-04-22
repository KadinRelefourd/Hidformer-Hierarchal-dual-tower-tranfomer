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

    def forward(self, x, mode="norm"):
        # x: (B, C, T)
        if mode == "norm":
            self.mean = x.mean(dim=-1, keepdim=True)
            self.std = x.std(dim=-1, keepdim=True)
            x_norm = (x - self.mean) / (self.std + self.eps)
            if self.affine:
                x_norm = x_norm * self.weight + self.bias
            return x_norm
        elif mode == "denorm":
            x_denorm = (x - (self.bias if self.affine else 0)) / (
                self.weight if self.affine else 1
            )
            return x_denorm * (self.std + self.eps) + self.mean
        else:
            raise ValueError("mode must be 'norm' or 'denorm'")


class TokenizeSequence(nn.Module):
    """
    Performs initial segmentation into tokens.
    """

    def __init__(self, token_length: int, stride: int):
        super().__init__()
        self.token_length = token_length
        self.stride = stride

    def forward(self, x):  # x: (B, C, T)
        B, C, T = x.shape
        # 1) segmentation -> (B, num_tokens, C, token_length)
        segments = x.unfold(
            -1, self.token_length, self.stride
        )  # (B, C, N, token_length)
        segments = segments.permute(0, 2, 1, 3).contiguous()  # (B, N, C, token_length)
        tokens = segments.view(B, segments.size(1), -1)  # (B, N, C*token_length)

        return tokens  # list of length num_blocks, each (B, Ni, token_dim)


class MergenceLayer(nn.Module):
    """Merges k-adjacent tokens into a single token.
    This is placed between sru++ blocks in the time tower.
    default: k=2
    """

    def __init__(
        self,
        k: int = 2,
        mode: str = "mean",  # {"mean", "linear"}
    ):
        super().__init__()
        assert k >= 1, "k must be positive"
        assert mode in {"mean", "linear"}, "mode must be 'mean' or 'linear'"
        self.k = k
        self.mode = mode
        if mode == "linear":
            self.proj: Optional[nn.Linear] = None  # initialised lazily on first call
        else:
            self.register_parameter("proj", None)

    # ------------------------------------------------------------------
    def forward(self, x: Tensor) -> Tensor:
        """Down‑sample sequence length by *k*.

        Parameters
        ----------
        x : Tensor of shape *(B, L, d)*.
        Returns
        -------
        Tensor
            Shape *(B, ⌈L/k⌉, d)* (mean) or *(B, ⌈L/k⌉, d)* (linear with learnable
            mixing across features).
        """
        B, L, D = x.shape
        if L % self.k != 0:
            pad_len = self.k - (L % self.k)
            pad_tensor = torch.zeros(B, pad_len, D, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad_tensor], dim=1)
            L = x.size(1)

        x = x.view(B, L // self.k, self.k, D)  # (B,L',k,D)

        if self.mode == "mean":
            return x.mean(dim=2)  # (B,L',D)

        # Lazy linear projection initialisation
        if self.proj is None:
            self.proj = nn.Linear(D * self.k, D, bias=True).to(x.device)
        x_flat = x.flatten(start_dim=2)  # (B,L',k*D)
        return self.proj(x_flat)


class SRUpp(nn.Module):
    """SRU++ block (Lei 2021) — GPU‑friendly recurrence with attention boost.

    The implementation follows the equations in the paper but avoids any custom
    CUDA kernels, keeping everything in plain PyTorch so that it works out of
    the box on both CPU & GPU (with automatic‑mixed precision, etc.).

    Input shape is *(B, L, d_in)* and outputs a *sequence* *(B, L, d_hid)* plus
    the final hidden state *(B, d_hid)* if requested.
    """

    def __init__(
        self,
        d_in: int,
        d_hidden: Optional[int] = None,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden or d_in
        self.dropout = dropout

        # Weight tying with single large projection (QKV style) for efficiency
        self.weight_proj = nn.Linear(self.d_in, self.d_hidden * 3, bias=bias)

        # Gate parameters (Element‑wise) — *vf*, *vr* in the paper.
        self.v_f = nn.Parameter(torch.zeros(self.d_hidden))
        self.v_r = nn.Parameter(torch.zeros(self.d_hidden))

        # Highway bias terms *bf*, *br*
        self.bias_f = nn.Parameter(torch.zeros(self.d_hidden))
        self.bias_r = nn.Parameter(torch.zeros(self.d_hidden))

        # Attention sub‑module to form *U* (eq. 12‑16) — lightweight variant.
        attn_dim = max(32, self.d_hidden // 4)
        self.q_proj = nn.Linear(self.d_in, attn_dim, bias=False)
        self.k_proj = nn.Linear(self.d_in, attn_dim, bias=False)
        self.v_proj = nn.Linear(self.d_in, attn_dim, bias=False)
        self.u_proj = nn.Linear(attn_dim, self.d_hidden * 3, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.5))  # residual mixing weight

    # ------------------------------------------------------------------
    def _compute_U(self, x: Tensor) -> Tensor:
        """Compute the aggregated tensor *U* (B, L, 3·H)."""
        Q, K, V = self.q_proj(x), self.k_proj(x), self.v_proj(x)  # (B,L,attn)
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
        attn = attn_logits.softmax(dim=-1)
        F_attn = torch.matmul(attn, V)  # (B,L,attn)
        mixed = self.alpha * F_attn + Q  # lightweight residual as in eq.16
        U = self.u_proj(mixed)  # (B,L,3·H)
        return U

    # ------------------------------------------------------------------
    def forward(self, x: Tensor, hidden_init: Optional[Tensor] = None) -> Tensor:
        """Run SRU++ over *x* and return the full output sequence.

        Parameters
        ----------
        x : Tensor
            Input of shape *(B, L, d_in)*.
        hidden_init : Tensor | None
            Optional initial hidden state *(B, d_hid)*. Default zeros.
        """
        B, L, _ = x.shape
        U_lin = self.weight_proj(x)  # (B,L,3·H) from linear path
        U_attn = self._compute_U(x)  # (B,L,3·H) from attention path
        U = U_lin + U_attn  # fuse paths

        # Gate splits -------------------------------------------------
        u_f, u_r, u_h = torch.chunk(U, 3, dim=-1)  # each (B,L,H)

        c_prev = (
            hidden_init
            if hidden_init is not None
            else torch.zeros(B, self.d_hidden, device=x.device, dtype=x.dtype)
        )
        h_seq = []
        dropout_mask = None
        if self.training and self.dropout > 0:
            dropout_mask = torch.dropout(
                torch.ones_like(c_prev), p=self.dropout, train=True
            )

        # Recurrence (loop over L — still fast due to fused matmuls) -------
        for t in range(L):
            f_t = torch.sigmoid(u_f[:, t] + self.v_f * c_prev + self.bias_f)
            c_t = f_t * c_prev + (1.0 - f_t) * u_h[:, t]
            r_t = torch.sigmoid(u_r[:, t] + self.v_r * c_prev + self.bias_r)
            h_t = r_t * c_t + (1.0 - r_t) * x[:, t]

            if dropout_mask is not None:
                h_t = h_t * dropout_mask

            h_seq.append(h_t)
            c_prev = c_t

        output = torch.stack(h_seq, dim=1)  # (B,L,H)
        return output  # *last hidden* can be c_prev if needed


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
        scores = torch.softmax(
            (Q @ K_hat.transpose(-1, -2)) / x.size(-1) ** 0.5, dim=-1
        )
        out = scores @ V_hat  # (B, N, k)
        # project back
        return nn.Linear(V_hat.size(-1), x.size(-1))(out)


class TimeBlock(nn.Module):
    def __init__(self, d_model, hidden_size):
        super().__init__()
        dropout = 0.2
        self.mixer = SRUPlusPlus(d_model, hidden_size)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, N, d_model)
        y = self.mixer(x)
        x = x + y  # moved the residual connection from before the normalization
        x = self.norm1(x)
        x = x + self.ffn(x)
        return self.norm2(x)


class FrequencyBlock(nn.Module):
    def __init__(self, d_model, k):
        super().__init__()
        self.mixer = LinearAttention(d_model, k)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
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
        out_dim=None,
    ):
        super().__init__()
        self.rev_in = RevIN(input_dim)
        self.segmenter = SegmentMerge(
            token_length, stride, max(time_blocks, freq_blocks)
        )
        token_dim = input_dim * token_length
        self.time_tower = nn.ModuleList(
            [TimeBlock(token_dim, hidden_size) for _ in range(time_blocks)]
        )
        self.freq_tower = nn.ModuleList(
            [FrequencyBlock(token_dim, freq_k) for _ in range(freq_blocks)]
        )
        # adaptors
        self.time_adapt = nn.Linear(token_dim, token_dim)
        self.freq_adapt = nn.Linear(token_dim, token_dim)
        # final decoder
        total_feats = token_dim * (time_blocks + freq_blocks)
        self.decoder = nn.Linear(total_feats, out_dim or input_dim)

    def forward(self, x):
        # x: (B, T, C) -> (B, C, T)
        x = x.transpose(1, 2)
        x_norm = self.rev_in(x, mode="norm")
        # segmentation + hierarchical outputs
        multi_tokens = self.segmenter(x_norm)  # list of length L: (B, Ni, token_dim)
        t_feats = []
        for tokens, block in zip(multi_tokens[: len(self.time_tower)], self.time_tower):
            out = block(tokens)  # (B, Ni, dim)
            t_feats.append(self.time_adapt(out.mean(dim=1)))
        # frequency tower: FFT then blocks
        fft_tokens = torch.fft.rfft(multi_tokens[0], dim=1).real
        f_feats = []
        for tokens, block in zip(
            [fft_tokens] + multi_tokens[1 : len(self.freq_tower)], self.freq_tower
        ):
            out = block(tokens)
            f_feats.append(self.freq_adapt(out.mean(dim=1)))
        # concat and decode
        all_feats = torch.cat(t_feats + f_feats, dim=-1)
        pred = self.decoder(all_feats)
        # denormalize -> (B, C, T')
        # Here pred is (B, C) for horizon prediction; adapt as needed
        return pred
