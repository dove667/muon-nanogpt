import torch
import torch.nn.functional as F
from torch import Tensor, nn


class RoPE(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        _, _, seq_len, head_dim = x.shape
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        cos = self.cos[:seq_len].to(dtype=x.dtype, device=x.device).view(1, 1, seq_len, head_dim // 2)
        sin = self.sin[:seq_len].to(dtype=x.dtype, device=x.device).view(1, 1, seq_len, head_dim // 2)
        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos
        return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)

class CausalSelfAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, head_dim: int, max_seq_len: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(model_dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, num_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * head_dim, model_dim, bias=False)
        self.rope = RoPE(head_dim, max_seq_len)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        q = self.rope(q)
        k = self.rope(k)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.head_dim)
        return self.out_proj(y)


class MLP(nn.Module):
    def __init__(self, model_dim: int, mlp_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(model_dim, mlp_dim, bias=False)
        self.fc2 = nn.Linear(mlp_dim, model_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, head_dim: int, mlp_dim: int, max_seq_len: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(model_dim)
        self.attn = CausalSelfAttention(model_dim, num_heads, head_dim, max_seq_len)
        self.ln2 = nn.LayerNorm(model_dim)
        self.mlp = MLP(model_dim, mlp_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        model_dim: int,
        max_seq_len: int,
        mlp_ratio: int = 4,
    ):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, model_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                model_dim=model_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                mlp_dim=mlp_ratio * model_dim,
                max_seq_len=max_seq_len,
            )
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(model_dim)
        self.lm_head = nn.Linear(model_dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight

        self.apply(self._init_weights)
        for name, param in self.named_parameters():
            param.label = name.replace(".weight", "")

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_seq: Tensor, target_seq: Tensor) -> Tensor:
        x = self.token_embed(input_seq)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            target_seq.reshape(-1),
            reduction="none",
        )
