
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from schedule import TrainingSchedule, default_training_stages, Hyperparameters


def relu_squared_mlp(x: Tensor, w1: Tensor, w2: Tensor) -> Tensor:
    x_flat = x.reshape(-1, x.shape[-1])
    hidden = F.relu(F.linear(x_flat, w1)).square()
    return (hidden @ w2).view_as(x)


def softcapped_cross_entropy(x: Tensor, target_seq: Tensor, weight: Tensor) -> Tensor:
    logits = x @ weight
    logits = 23 * torch.sigmoid((logits + 5) / 7.5)
    return F.cross_entropy(logits.float(), target_seq, reduction="none")

args = None
device = None


def setup_model_runtime(*, args_value, grad_accum_steps_value, device_value):
    global args, device
    args = args_value
    device = device_value


def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))


class CastedLinearT(nn.Module):
    """Linear layer with explicit transposed weight storage."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(in_features, out_features, dtype=torch.bfloat16))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            nn.init.zeros_(self.weight)

    def forward(self, x: Tensor):
        return x @ self.weight.type_as(x)

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the model

class Yarn(nn.Module):
    def __init__(self, head_dim, max_seq_len, paired=False):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.paired = paired
        self.reset()

    def rotary(self, x_BTHD):
        assert self.factor1.size(0) >= x_BTHD.size(-3)
        factor1, factor2 = (
            self.factor1[None, : x_BTHD.size(-3), None, :],
            self.factor2[None, : x_BTHD.size(-3), None, :],
        )
        x_flip = x_BTHD.view(*x_BTHD.shape[:-1], x_BTHD.shape[-1] // 2, 2).flip(-1).view(x_BTHD.shape)
        return factor1 * x_BTHD + factor2 * x_flip

    def reset(self):
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=self.head_dim//4, dtype=torch.float32, device=device)
        angular_freq = angular_freq.repeat_interleave(2)
        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(self.head_dim//2)])
        t = torch.arange(2*self.max_seq_len, dtype=torch.float32, device=device)
        if not self.paired:
            theta = torch.outer(t, angular_freq)
            self.factor1 = nn.Buffer(
                theta.cos().to(torch.bfloat16), persistent=False
            )
            self.factor2 = nn.Buffer(
                theta.sin().to(torch.bfloat16), persistent=False
            )
        else:
            t_even = 2 * t
            t_odd = t_even + 1
            theta1 = torch.outer(t_even, angular_freq)
            theta2 = torch.outer(t_odd, angular_freq)
            self.factor1 = nn.Buffer(
                torch.cat((theta1.cos(), theta2.cos()), dim=-1).to(torch.bfloat16),
                persistent=False
            )
            self.factor2 = nn.Buffer(
                torch.cat((theta1.sin(), theta2.sin()), dim=-1).to(torch.bfloat16),
                persistent=False
            )
        self.factor2[..., 1::2] *= -1
        self.angular_freq = angular_freq
        # start with 0.1, inspired by 0.12 from @leloykun and learnable scalars used by @brendanh0gan https://x.com/hi_tysam/status/1879693583898591283
        self.attn_scale = 0.1

    def apply(self, old_window: int, new_window: int, alpha: int=1, beta: int=32):
        rotations = old_window * self.angular_freq / (2 * torch.pi)
        scaling_factor = old_window / new_window
        interpolation_weight = torch.clamp((rotations - alpha) / (beta - alpha), 0, 1)
        self.angular_freq *= scaling_factor + interpolation_weight * (1 - scaling_factor)
        t = torch.arange(2*self.max_seq_len, dtype=torch.float32, device=self.angular_freq.device)
        if not self.paired:
            theta = torch.outer(t, self.angular_freq)
            self.factor1.copy_(theta.cos())
            self.factor2.copy_(theta.sin())
        else:
            t_even = 2 * t
            t_odd = t_even + 1
            theta1 = torch.outer(t_even, self.angular_freq)
            theta2 = torch.outer(t_odd, self.angular_freq)
            self.factor1.copy_(torch.cat((theta1.cos(), theta2.cos()), dim=-1))
            self.factor2.copy_(torch.cat((theta1.sin(), theta2.sin()), dim=-1))
        self.factor2[..., 1::2] *= -1
        self.attn_scale *= 0.2 * math.log(new_window / old_window) + 1

@dataclass(slots=True)
class AttnArgs:
    ve: torch.Tensor
    sa_lambdas: torch.Tensor
    seqlens: torch.Tensor
    bm_size: int
    yarn: Yarn
    key_offset: bool
    attn_gate_w: torch.Tensor
    ve_gate_w: torch.Tensor
    train_max_seq_len: torch.Tensor


class _CudaSdpaFlashAttnInterface:
    @staticmethod
    def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                               causal=True, softmax_scale=None, window_size=(-1, -1), **_):
        assert cu_seqlens_q is cu_seqlens_k or torch.equal(cu_seqlens_q, cu_seqlens_k)
        cu = cu_seqlens_q.detach().cpu().tolist()
        left_window = -1 if window_size is None else int(window_size[0])
        outputs = []
        for start, end in zip(cu[:-1], cu[1:]):
            if end <= start:
                continue
            qq = q[start:end].transpose(0, 1).unsqueeze(0)
            kk = k[start:end].transpose(0, 1).unsqueeze(0)
            vv = v[start:end].transpose(0, 1).unsqueeze(0)
            seq_len = end - start
            use_window = left_window >= 0 and left_window < seq_len
            if use_window:
                idx = torch.arange(seq_len, device=q.device)
                keep = (idx[None, :] <= idx[:, None]) & (idx[None, :] >= idx[:, None] - left_window)
                attn_mask = torch.zeros((seq_len, seq_len), device=q.device, dtype=q.dtype)
                attn_mask.masked_fill_(~keep, float("-inf"))
                out = F.scaled_dot_product_attention(
                    qq, kk, vv, attn_mask=attn_mask, dropout_p=0.0,
                    is_causal=False, scale=softmax_scale,
                )
            else:
                out = F.scaled_dot_product_attention(
                    qq, kk, vv, dropout_p=0.0, is_causal=causal, scale=softmax_scale,
                )
            outputs.append(out.squeeze(0).transpose(0, 1))
        return torch.cat(outputs, dim=0)

flash_attn_interface = _CudaSdpaFlashAttnInterface()


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim: int, num_heads: int, paired: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dim = dim
        self.hdim = num_heads * head_dim
        self.paired = paired
        assert self.hdim == self.dim, "num_heads * head_dim must equal model_dim"
        # Weights are stored in parameter banks and passed via forward()

    def forward(self, x: Tensor, attn_args: AttnArgs, qkvo_w: Tensor):
        B, T = x.size(0), x.size(1) # batch size, sequence length
        assert B == 1, "varlen sequences requires B == 1"
        assert T % 16 == 0
        # unpack attention args
        yarn = attn_args.yarn
        ve, sa_lambdas, key_offset = attn_args.ve, attn_args.sa_lambdas, attn_args.key_offset
        seqlens, bm_size = attn_args.seqlens, attn_args.bm_size
        # sparse gated attention to enable context based no-op by @classiclarryd
        # only include gates on layers with value embeds used on forward pass
        attn_gate_w, ve_gate_w = attn_args.attn_gate_w, attn_args.ve_gate_w
        train_max_seq_len = attn_args.train_max_seq_len

        q, k, v = F.linear(x, sa_lambdas[0] * qkvo_w[:self.dim * 3].type_as(x)).view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
        max_len = train_max_seq_len if self.training else args.val_batch_size

        q, k = norm(q), norm(k) # QK norm @Grad62304977

        if not self.paired:
            q, k = yarn.rotary(q), yarn.rotary(k)

            if key_offset:
                # shift keys forward for the stationary head dims. Enables 1-layer induction.
                k[:, 1:, :, self.head_dim // 2:] = k[:, :-1, :, self.head_dim // 2:]

            if ve is not None:
                # gate pattern g(x[:6] + ve[:6]) by @photomz
                ve_gate_out = 2 * torch.sigmoid(F.linear(torch.cat([x[..., :6], ve[None, ..., :6]], dim=-1), ve_gate_w)).view(B, T, self.num_heads, 1)
                v = v + ve_gate_out * ve.view_as(v) # @ KoszarskyB & @Grad62304977

        else:
            # Paired heads: adjacent heads' queries attend to each other's keys.
            # Two copies of the input stream are interleaved to achieve this, which:
            # - doubles the length of each sequence
            # - halves the effective window size
            q = q.view(B, T, self.num_heads // 2, self.head_dim * 2)
            k = k.view(B, T, self.num_heads // 2, self.head_dim * 2)
            v = v.reshape(B, T * 2, self.num_heads // 2, self.head_dim)

            q, k = yarn.rotary(q), yarn.rotary(k)

            q = q.view(B, T * 2, self.num_heads // 2, self.head_dim)
            k = k.view(B, T * 2, self.num_heads // 2, self.head_dim)

            if ve is not None:
                ve_gate_out = 2 * torch.sigmoid(F.linear(x[..., :12], ve_gate_w)).view(B, T * 2, self.num_heads // 2, 1)
                v = v + ve_gate_out * ve.view_as(v)

            seqlens = 2 * seqlens
            max_len = 2 * max_len

        # use flash_attn over flex_attn @varunneal. flash_attn_varlen suggested by @YouJiacheng
        y = flash_attn_interface.flash_attn_varlen_func(q[0], k[0], v[0], cu_seqlens_q=seqlens, cu_seqlens_k=seqlens,
                                                        max_seqlen_q=max_len, max_seqlen_k=max_len,
                                                        causal=True, softmax_scale=yarn.attn_scale, window_size=(bm_size, 0))
        y = y.view(B, T, self.num_heads, self.head_dim)
        y = y * torch.sigmoid(F.linear(x[..., :12], attn_gate_w)).view(B, T, self.num_heads, 1)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim) # re-assemble all head outputs side by side
        y = F.linear(y, sa_lambdas[1] * qkvo_w[self.dim * 3:].type_as(y))  # sa_lambdas[1] pre-multiplied to O @shenberg
        return y


# -----------------------------------------------------------------------------
# The main model

def next_multiple_of_n(v: float | int, *, n: int):
    return math.ceil(v / n) * n

@dataclass(slots=True)
class ForwardScheduleConfig:
    mtp_weights: torch.Tensor
    ws_short: int
    ws_long: int
    train_max_seq_len: int

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, head_dim: int, model_dim: int, max_seq_len: int):
        super().__init__()
        self.num_layers = num_layers
        self.vocab_size = next_multiple_of_n(vocab_size, n=128)

        self.smear_gate = nn.Linear(12, 1, bias=False)
        nn.init.zeros_(self.smear_gate.weight)

        self.skip_gate = nn.Linear(12, 1, bias=False)
        nn.init.zeros_(self.skip_gate.weight)

        # token value embeddings by @KoszarskyB - inspired by @Grad62304977's value residual implementation following https://arxiv.org/abs/2410.17897
        # value embedding code simplification inspired by @ragulpr https://github.com/KellerJordan/modded-nanogpt/pull/78
        # spherical gaussian init by @photomz
        self.value_embeds = nn.Parameter(0.01 * torch.randn(5 * self.vocab_size, model_dim, dtype=torch.bfloat16))

        # parameter banks for attention and value embedding gate weights
        self.attn_gate_bank = nn.Parameter(torch.zeros(10, num_heads, 12)) # 10 layers
        self.ve_gate_bank = nn.Parameter(torch.zeros(5, num_heads, 12)) # 5 unique gates
        self.gate_filler_nones = [None] * (num_layers - 6)

        # -----------------------------------
        # Parameter banks for sharded optimization, by @chrisjmccormick

        # Identify which layers have attention/MLP
        # Attention is skipped in layer 6 by @YouJiacheng
        num_attn_layers = num_layers - 1
        # All layers have MLP (At 11 layers--dropped first layer @EmelyanenkoK)
        num_mlp_layers = num_layers

        hdim = num_heads * head_dim
        mlp_hdim = 4 * model_dim

        # QK bank: per-head-pair Muon groups for Q, K weights
        # Each pair of adjacent heads gets its own independent polar express orthogonalization
        self._num_attn_layers = num_attn_layers
        num_qk_groups = num_attn_layers * 2 * (num_heads // 2)  # 10 * 2 * 3 = 60
        self._num_qk_groups = num_qk_groups
        self.qk_bank = nn.Parameter(torch.empty(num_qk_groups, head_dim * 2, model_dim))
        self.qk_bank.reshape = (num_qk_groups, head_dim * 2, model_dim)

        # VO bank: per-layer Muon groups for V and O weights
        num_vo_real = num_attn_layers * 2  # 20
        self.vo_bank = nn.Parameter(torch.empty(num_vo_real, hdim, hdim))
        self.vo_bank.reshape = (num_vo_real, hdim, hdim)

        # MLP bank: stores c_fc and c_proj for all MLP layers
        # We add 1 padding layer (index 11) to get 12*2=24 matrices for even distribution across 8 GPUs
        self.mlp_bank = nn.Parameter(torch.empty(12, 2, mlp_hdim, model_dim))  # (12, 2, 3072, 768)
        self.mlp_bank.reshape = (24, mlp_hdim, model_dim)  # Shape for sharding: (24, 3072, 768)

        # improved init scale by @YouJiacheng and @srashedll
        std = 0.5 * model_dim ** -0.5
        bound = (3 ** 0.5) * std
        with torch.no_grad():
            self.qk_bank[:num_qk_groups].uniform_(-bound, bound)
            self.qk_bank[num_qk_groups:].zero_()
            self.vo_bank[:num_vo_real].uniform_(-bound, bound)
            self.vo_bank[num_vo_real:].zero_()
            self.mlp_bank[:, 0, :, :].uniform_(-bound, bound)  # c_fc
            self.mlp_bank[:, 1, :, :].zero_()  # c_proj - zero init suggested by @Grad62304977

        # Attention modules (no learned params -- weights come from qk_bank/vo_bank)
        self.paired_head_layers = [0, 2, 5, 9]
        self.attn = CausalSelfAttention(model_dim, head_dim, num_heads, paired=False)
        self.attn_paired = CausalSelfAttention(model_dim, head_dim, num_heads, paired=True)
        self.yarn = Yarn(head_dim, max_seq_len)
        self.yarn_paired_head = Yarn(head_dim, max_seq_len, paired=True)
        # there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency.
        # suggested to me by @Grad62304977. this originates from Karpathy's experiments.
        self.lm_head = CastedLinearT(model_dim, self.vocab_size)

        nn.init.normal_(self.lm_head.weight, mean=0, std=0.005)

        self.embed = nn.Embedding(self.vocab_size, model_dim)
        with torch.no_grad():
            self.embed.weight.copy_(self.lm_head.weight.T)

        self.bigram_embed = nn.Embedding(args.bigram_vocab_size, model_dim)
        nn.init.zeros_(self.bigram_embed.weight)

        self.post_lambdas = nn.Parameter(torch.ones(num_layers, 2))

        # Per-layer injection coefficients for x0 and bigram
        self.x0_lambdas = nn.Parameter(torch.zeros(num_layers))
        self.bigram_lambdas = nn.Parameter(0.05 * torch.ones(num_layers))

        # Per-sublayer residual scaling: [num_layers, 2] where [:,0]=attn, [:,1]=mlp
        # sqrt(1.1) per sublayer so cumulative per-layer scaling is 1.1
        self.resid_lambdas = nn.Parameter(torch.full((num_layers, 2), 1.1**0.5))

        pad = 0
        self.scalars = nn.Parameter(
            torch.cat(
                [
                    *[torch.tensor([0.5, 1.0]) for _ in range(num_layers)],  # SA lambdas
                    torch.zeros(1), # smear_lambda
                    0.5*torch.ones(1), # backout_lambda
                    -1.5 * torch.ones(1),  # skip_lambda -> σ(-1.5) ≈ 0.18
                    torch.ones(pad),
                ]
            )
        )
        # Auto-label parameters
        for name, param in self.named_parameters():
            param.label = name.replace('.weight', '')

    def forward(self, input_seq: Tensor, target_seq: Tensor, seqlens: Tensor, bigram_input_seq: Tensor, schedule_cfg: ForwardScheduleConfig):
        assert input_seq.ndim == 1

        # ---- Schedule and layer topology ----
        mtp_weights, train_max_seq_len = schedule_cfg.mtp_weights, schedule_cfg.train_max_seq_len
        ws_short, ws_long = schedule_cfg.ws_short, schedule_cfg.ws_long
        # set block masks and key shift
        bm_sizes = [ws_short, ws_short, ws_short, ws_long, ws_short, ws_short, None, ws_short, ws_short, ws_short, ws_long]
        assert len(bm_sizes) == self.num_layers
        key_offset = [b==ws_long for b in bm_sizes] # apply partial key offset to long windows

        # ---- Unbind parameters (avoid select_backward kernels) ----
        sa_lambdas = self.scalars[: 2 * self.num_layers].view(-1, 2)
        smear_lambda = self.scalars[2 * self.num_layers]
        backout_lambda = self.scalars[2 * self.num_layers + 1]
        skip_lambda = self.scalars[2 * self.num_layers + 2]
        resid_lambdas_attn = self.resid_lambdas[:, 0].bfloat16().unbind(0)
        resid_lambdas_mlp  = self.resid_lambdas[:, 1].bfloat16().unbind(0)
        post_lambdas_attn = self.post_lambdas[:, 0].bfloat16().unbind(0)
        post_lambdas_mlp  = self.post_lambdas[:, 1].bfloat16().unbind(0)
        x0_lambdas = self.x0_lambdas.bfloat16().unbind(0)
        bigram_lambdas = self.bigram_lambdas.bfloat16().unbind(0)
        ag = self.attn_gate_bank.unbind(0)
        veg = self.ve_gate_bank.unbind(0)
        attn_gates = [*ag[:6], None, *ag[6:]]
        ve_gates = [None, veg[0], veg[1], *self.gate_filler_nones, veg[2], veg[3], veg[4]]
        assert len(attn_gates) == self.num_layers
        assert len(ve_gates) == self.num_layers
        qk_all = self.qk_bank[:self._num_qk_groups].view(self._num_attn_layers, -1, self.qk_bank.shape[-1])
        vo_flat = self.vo_bank[:self._num_attn_layers * 2].view(self._num_attn_layers, 2, *self.vo_bank.shape[1:]).flatten(1, 2)
        attn_weights = torch.cat([qk_all, vo_flat], dim=1).unbind(0)
        mlp_all = self.mlp_bank.flatten(0, 1).unbind(0)  # 24 tensors of [mlp_hdim, dim]
        mlp_fcs = mlp_all[0::2]    # even indices: c_fc
        mlp_projs = mlp_all[1::2]  # odd indices: c_proj

        # ---- Embeddings and input preparation ----
        x = self.embed(input_seq) # embed is synced from lm_head during tied phase by optimizer
        
        x0_bigram = self.bigram_embed(bigram_input_seq)[None]

        # Value embeddings - always computed (not precomputed)
        ve = self.value_embeds.view(5, self.vocab_size, -1)[:, input_seq]
        # Shifted .01 ... 234 structure on token value embeddings by @photomz
        ve = [None, ve[0], ve[1], *self.gate_filler_nones, ve[2], ve[3], ve[4]]
        assert len(ve) == self.num_layers

        # smear token embed forward 1 position @classiclarryd
        smear_gate_out = smear_lambda * torch.sigmoid(self.smear_gate(x[1:, :self.smear_gate.weight.size(-1)]))
        x = torch.cat([x[:1], x[1:] + smear_gate_out * x[:-1]])
        x = x0 = norm(x[None])

        # Initialize residual stream with pre-layer-0 bigram injection
        x = x + x0_bigram * bigram_lambdas[0]

        # Precompute x0/bigram injection (added to attention output each layer)
        # Layer 0: bigram already injected above, so only x0 component
        x0_inject = (x0 * x0_lambdas[0],) + tuple(x0 * x0_lambdas[i] + x0_bigram * bigram_lambdas[i] for i in range(1, self.num_layers))
        skip_gate_out = torch.sigmoid(skip_lambda) * 2 * torch.sigmoid(self.skip_gate(x0[..., :self.skip_gate.weight.size(-1)]))
        
        # ---- Transformer layers ----
        x_backout = None
        skip_connection = None
        for i in range(self.num_layers):
            yarn = self.yarn_paired_head if i in self.paired_head_layers else self.yarn
            attn_args = AttnArgs(
                ve=ve[i],
                sa_lambdas=sa_lambdas[i],
                seqlens=seqlens,
                bm_size=bm_sizes[i],
                yarn=yarn,
                key_offset=key_offset[i],
                attn_gate_w=attn_gates[i],
                ve_gate_w=ve_gates[i],
                train_max_seq_len=train_max_seq_len
            )
            # Select weights from banks
            attn_idx = i - (i > 6) if i != 6 else None
            qkvo_w = attn_weights[attn_idx] if attn_idx is not None else None
            c_fc = mlp_fcs[i]
            c_proj = mlp_projs[i]

            # Select attention variant for this layer
            attn = self.attn_paired if i in self.paired_head_layers else self.attn

            # Skip attention on layer 6 @YouJiacheng. Instead pull skip connection from prior long window
            if i == 6:
                x = x + skip_gate_out * skip_connection
            else:
                attn_in = x_backout if x_backout is not None else x
                attn_out = attn(norm(attn_in), attn_args, qkvo_w)
                x = resid_lambdas_attn[i] * x + post_lambdas_attn[i] * attn_out + x0_inject[i]
            x = resid_lambdas_mlp[i] * x + post_lambdas_mlp[i] * relu_squared_mlp(norm(x), c_fc, c_proj)
            if i == 3:
                skip_connection = x
            if i == 7:
                x_backout = x

        # back out contributions from first 7 layers
        x -= backout_lambda * x_backout
        x = norm(x)
        # @Grad62304977 added tanh softcapping following Gemma 2 paper, @KoszarskyB reduced it from 30 to 15
        # @YouJiacheng shifted it by +15 (2*sigmoid(2*x)=tanh(x)+1). @classiclarryd updated to 23*sigmoid((logits+5)/7.5)
        if self.training:
            loss_per_token = softcapped_cross_entropy(x.view(-1, x.size(-1)), target_seq, self.lm_head.weight)
        else:
            logits = self.lm_head(x)
            logits = 23 * torch.sigmoid((logits + 5) / 7.5)
            logits_for_loss = logits.float()
            loss_per_token = F.cross_entropy(logits_for_loss.view(-1, logits_for_loss.size(-1)), target_seq, reduction="none")
        return loss_per_token
