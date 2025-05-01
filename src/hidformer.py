import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
import math


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
        bias: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden or d_in
        self.dropout = dropout

        # Weight tying with single large projection (QKV style) for efficiency
        # self.weight_proj = nn.Linear(self.d_in, self.d_hidden * 3, bias=bias)

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
        # U_lin = self.weight_proj(x)  # (B,L,3·H) from linear path
        U_attn = self._compute_U(x)  # (B,L,3·H) from attention path
        # U = U_lin + U_attn  # fuse paths
        U = U_attn  # use only attention path

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
        self.k = k
        self.d_model = d_model
        self.to_q = nn.Linear(d_model, d_model, bias=False)
        self.to_k = nn.Linear(d_model, d_model, bias=False)
        self.to_v = nn.Linear(d_model, d_model, bias=False)
        self.A = nn.Parameter(torch.randn(d_model, k))
        self.proj_out = nn.Linear(k, d_model, bias=False)
        # share A for B
        self.B = self.A
        self.C = nn.Parameter(torch.randn(d_model, k))

    def forward(self, x):  # x: (B, N, d_model)
        B, N, D = x.shape
        Q = self.to_q(x)  # (B, N, d)
        K = self.to_k(x)
        V = self.to_v(x)
        Q_hat = Q @ self.C  # (B, N, k)
        K_hat = K @ self.A  # (B, N, k)
        V_hat = V @ self.B  # (B, N, k)
        attn_scores = torch.softmax(
            (Q_hat @ K_hat.transpose(-1, -2)) / math.sqrt(self.k),  # Use k for scaling
            dim=-1,
        )
        out = attn_scores @ V_hat  # (B, N, k)
        # project back
        return self.proj_out(out)


class TimeBlock(nn.Module):
    def __init__(self, d_model, hidden_size, dropout=0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.mixer = SRUpp(d_model, hidden_size, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (B, N, d_model)
        x_norm = self.norm1(x)
        x = self.mixer(x_norm) + x
        x_norm = self.norm2(x)
        x = x + self.ffn(x_norm)
        return x


class FrequencyBlock(nn.Module):
    def __init__(self, d_model, k, dropout=0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.mixer = LinearAttention(d_model, k)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x_norm = self.norm1(x)
        x = self.mixer(x_norm) + x
        x_norm = self.norm2(x)
        x = self.ffn(x_norm) + x

        return x


class Hidformer(nn.Module):
    """
    Hidformer model implementing hierarchical dual towers with multi-scale mergence.
    Based on Liu et al., 2024. Uses Pre-Normalization blocks.
    """

    def __init__(
        self,
        input_dim: int,  # Number of input features (C)
        pred_len: int,  # Prediction horizon (H)
        token_length: int = 16,  # Length of each token segment (T_token)
        stride: int = 8,  # Stride for token segmentation (S)
        num_time_blocks: int = 4,  # Number of blocks in Time Tower
        num_freq_blocks: int = 2,  # Number of blocks in Frequency Tower
        d_model: int = 128,  # Main dimension of the model's hidden states
        # time_hidden_factor: int = 1, # SRUpp hidden dim factor (removed, assume d_hidden=d_model)
        freq_k: int = 64,  # Low-rank dimension for LinearAttention
        dropout: float = 0.2,  # Dropout rate
        merge_mode: str = "linear",  # Mode for MergenceLayer ('linear' or 'mean')
        merge_k: int = 2,  # How many tokens to merge at each step
    ):
        super().__init__()
        self.pred_len = pred_len
        self.input_dim = input_dim
        self.d_model = d_model

        # 1. Normalization
        self.rev_in = RevIN(input_dim)  # Operates on (B, C, T_in)

        # 2. Initial Tokenization
        self.segmenter = TokenizeSequence(token_length, stride)
        # Output: (B, N, C*token_length)
        initial_token_dim = input_dim * token_length

        # 3. Input Projection (Project initial tokens to d_model)
        self.input_proj = nn.Linear(initial_token_dim, d_model)

        # --- Towers ---
        # Use separate ModuleLists for blocks and mergers
        self.time_blocks = nn.ModuleList()
        self.time_mergers = nn.ModuleList()
        for i in range(num_time_blocks):
            # TimeBlock uses d_model for input and hidden size
            self.time_blocks.append(
                TimeBlock(d_model=d_model, hidden_size=d_model, dropout=dropout)
            )
            if i < num_time_blocks - 1:  # Add merger except after last block
                self.time_mergers.append(MergenceLayer(k=merge_k, mode=merge_mode))

        self.freq_blocks = nn.ModuleList()
        self.freq_mergers = nn.ModuleList()
        # Projection layer for FFT output before feeding to Frequency Tower
        # FFT output size depends on input length T_in. Let's make it adaptive later.
        # For now, assume we project FFT tokens back to d_model.
        self.fft_proj = None  # Will initialize lazily in forward

        for i in range(num_freq_blocks):
            self.freq_blocks.append(
                FrequencyBlock(d_model=d_model, k=freq_k, dropout=dropout)
            )
            if i < num_freq_blocks - 1:
                self.freq_mergers.append(MergenceLayer(k=merge_k, mode=merge_mode))

        # --- Decoder ---
        # Following Fig 4: Adapt final tower outputs, concatenate, then decode
        self.time_adaptor = nn.Linear(
            d_model, d_model
        )  # Adapts final time tower output
        self.freq_adaptor = nn.Linear(
            d_model, d_model
        )  # Adapts final freq tower output
        decoder_input_dim = d_model * 2  # After adaptation and concatenation

        # Final linear head for prediction
        # Predicts pred_len steps for each of the input_dim features
        self.decoder = nn.Linear(decoder_input_dim, decoder_input_dim)
        self.decoder2 = nn.Linear(decoder_input_dim, self.pred_len * input_dim)

    def forward(self, x_enc: torch.Tensor) -> torch.Tensor:
        # x_enc: (B, T_in, C) - Input historical data
        B, T_in, C = x_enc.shape
        if C != self.input_dim:
            raise ValueError(
                f"Input feature dim {C} != init input_dim {self.input_dim}"
            )

        # 1. RevIN Normalization
        x_enc_transposed = x_enc.transpose(1, 2)  # (B, C, T_in)
        x_norm = self.rev_in(x_enc_transposed, mode="norm")  # (B, C, T_in)

        # --- Time Tower Path ---
        # 2. Initial Tokenization for Time Tower
        time_tokens = self.segmenter(x_norm)  # (B, N, C*token_length)
        # 3. Project to d_model
        time_tokens = self.input_proj(time_tokens)  # (B, N, d_model)

        # 4. Process through Time Tower blocks and mergers
        current_time_tokens = time_tokens
        time_block_outputs = (
            []
        )  # Store output of each block if needed for other agg strategies
        for i, block in enumerate(self.time_blocks):
            processed_tokens = block(current_time_tokens)  # (B, Ni, d_model)
            time_block_outputs.append(processed_tokens)  # Store output
            if i < len(self.time_mergers):
                current_time_tokens = self.time_mergers[i](processed_tokens)
            else:
                final_time_output = processed_tokens  # Output of the last block

        # --- Frequency Tower Path ---
        # 1. Apply FFT to the *normalized input sequence* (more standard)
        fft_output = torch.fft.rfft(x_norm, dim=-1)  # (B, C, T_freq)
        # Use magnitude and phase or just magnitude? Let's use magnitude for simplicity.
        # Could concatenate real/imag parts as features too.
        fft_features = fft_output.abs()  # (B, C, T_freq)

        # 2. Tokenize the frequency features
        # Need to handle potential dimension mismatch if T_freq is different
        # For simplicity, let's assume TokenizeSequence can handle (B, C, T_freq)
        # This might require adjusting stride/token_length for frequency domain
        # Or pad/truncate T_freq to match T_in expectations?
        # Alternative: Use a different tokenizer or adapt FrequencyBlocks.

        # --- Let's try tokenizing frequency features directly ---
        # This assumes token_length and stride are meaningful for frequency domain.
        # May need separate token_length_freq, stride_freq parameters.
        # For now, reuse time tokenizer settings.
        try:
            freq_tokens = self.segmenter(
                fft_features
            )  # (B, N_freq, C*token_length_freq)
        except ValueError as e:
            # Handle cases where T_freq < token_length
            print(
                f"Warning: Could not tokenize frequency features due to length. Skipping Freq Tower. Error: {e}"
            )
            # Create a dummy zero tensor for freq path output if needed
            final_freq_output = torch.zeros_like(final_time_output)  # Match shape
            freq_tokens_proj = torch.zeros_like(time_tokens)  # Match shape
        else:
            # 3. Project frequency tokens to d_model
            # Lazy initialization of FFT projection layer
            if self.fft_proj is None:
                fft_token_dim = freq_tokens.shape[-1]
                self.fft_proj = nn.Linear(fft_token_dim, self.d_model).to(x_enc.device)
            freq_tokens_proj = self.fft_proj(freq_tokens)  # (B, N_freq, d_model)

            # 4. Process through Frequency Tower blocks and mergers
            current_freq_tokens = freq_tokens_proj
            freq_block_outputs = []
            for i, block in enumerate(self.freq_blocks):
                processed_tokens = block(current_freq_tokens)  # (B, Ni_freq, d_model)
                freq_block_outputs.append(processed_tokens)
                if i < len(self.freq_mergers):
                    current_freq_tokens = self.freq_mergers[i](processed_tokens)
                else:
                    final_freq_output = processed_tokens  # Output of the last block

        # --- Aggregation & Decoder (Following Fig 4) ---
        # Aggregate final outputs (e.g., mean pool over sequence dim N)
        # Ensure outputs exist (handle skipped freq tower case)
        if "final_time_output" not in locals():
            raise RuntimeError("Time tower did not produce an output.")
        if "final_freq_output" not in locals():
            # Handle case where freq tower was skipped
            print("Warning: Frequency tower output missing, using zeros.")
            final_freq_output = torch.zeros_like(final_time_output)

        # Use mean aggregation over the token sequence dimension
        agg_time = final_time_output.mean(dim=1)  # (B, d_model)
        agg_freq = final_freq_output.mean(dim=1)  # (B, d_model)

        # Adapt aggregated outputs
        adapted_time = self.time_adaptor(agg_time)  # (B, d_model)
        adapted_freq = self.freq_adaptor(agg_freq)  # (B, d_model)

        # Concatenate adapted features
        combined_features = torch.cat(
            [adapted_time, adapted_freq], dim=-1
        )  # (B, d_model * 2)

        # Final Decoder
        pred = self.decoder(combined_features)  # (B, d_model * 2)
        pred = self.decoder2(pred)  # (B, pred_len * input_dim)

        # Reshape prediction: (B, pred_len, input_dim)
        pred = pred.view(B, self.pred_len, self.input_dim)

        # pred = pred.view(B, self.pred_len, 1)

        # Transpose for RevIN: (B, input_dim, pred_len)
        pred = pred.transpose(1, 2)

        # 5. Denormalize Output
        pred_denorm = self.rev_in(pred, mode="denorm")  # (B, C, pred_len)

        # Final output shape: (B, pred_len, C)
        return pred_denorm.transpose(1, 2)
