from typing import Any
import torch
import torch.nn as nn


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))


class FeedForwardNetwork(nn.Module):

    def __init__(self, cfg: Any):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg.em_dim, 4 * cfg.em_dim),
            GELU(),
            nn.Linear(4 * cfg.em_dim, cfg.em_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class MultiHeadAttention(nn.Module):
    def __init__(
        self, d_in: int, d_out: int, num_heads: int, context_length: int,
        dropout: float = 0.0, qkv_bias: bool = False
    ):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, num_tokens, _ = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        queries = queries.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)

        context_vec = self.out_proj(context_vec)
        return context_vec


class TransformerBlock(nn.Module):
    def __init__(self, cfg: Any):
        super().__init__()
        self.attention = MultiHeadAttention(
            d_in=cfg.em_dim,
            d_out=cfg.em_dim,
            context_length=cfg.context_length,
            num_heads=cfg.n_heads,
            dropout=cfg.drop_rate,
            qkv_bias=cfg.qkv_bias
        )
        self.ff = FeedForwardNetwork(cfg)
        self.norm1 = LayerNorm(cfg.em_dim)
        self.norm2 = LayerNorm(cfg.em_dim)
        self.drop_shortcut = nn.Dropout(cfg.drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        x = self.norm1(x)
        x = self.attention(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x


class LayerNorm(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GPTModel(nn.Module):

    def __init__(self, config: Any):
        super().__init__()
        self.cfg = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.em_dim)
        self.pos_emb = nn.Embedding(config.context_length, config.em_dim)
        self.drop_embs = nn.Dropout(config.drop_rate)
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.final_norm = LayerNorm(config.em_dim)
        self.out_head = nn.Linear(config.em_dim, config.vocab_size, bias=False)

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:

        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_embs(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
