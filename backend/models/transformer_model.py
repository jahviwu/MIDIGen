import math
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint   # <-- added

# Updated default config
DEFAULT_CONFIG = {
    "vocab_size":   404,    # must match vocab.json
    "d_model":      128,
    "n_heads":      4,
    "n_layers":     4,
    "d_ff":         512,
    "max_seq_len":  1024,
    "dropout":      0.1,
    "pad_token_id": 0,
}

# Positional Encoding
class SinusoidalPositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)

# Single Decoder Block
class TransformerDecoderBlock(nn.Module):

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn  = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ff         = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        residual = x
        x = self.norm1(x)
        attn_out, _ = self.self_attn(
            x, x, x,
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = residual + attn_out

        residual = x
        x = self.norm2(x)
        x = residual + self.ff(x)
        return x

class MusicTransformer(nn.Module):

    def __init__(self, config: dict | None = None):
        super().__init__()
        cfg = {**DEFAULT_CONFIG, **(config or {})}
        self.cfg = cfg

        # Enable gradient checkpointing
        self.gradient_checkpointing = True

        self.embedding = nn.Embedding(
            cfg["vocab_size"], cfg["d_model"], padding_idx=cfg["pad_token_id"]
        )
        self.pos_enc = SinusoidalPositionalEncoding(
            cfg["d_model"], cfg["max_seq_len"], cfg["dropout"]
        )
        self.blocks = nn.ModuleList([
            TransformerDecoderBlock(
                cfg["d_model"], cfg["n_heads"], cfg["d_ff"], cfg["dropout"]
            )
            for _ in range(cfg["n_layers"])
        ])
        self.norm_out = nn.LayerNorm(cfg["d_model"])
        self.head     = nn.Linear(cfg["d_model"], cfg["vocab_size"], bias=False)

        # Share embedding and output projection weights
        self.head.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        B, T = input_ids.shape
        causal_mask = self._causal_mask(T, input_ids.device)

        x = self.embedding(input_ids)
        x = self.pos_enc(x)

        # Gradient checkpointing applied here
        for block in self.blocks:
            if self.gradient_checkpointing:
                x = checkpoint.checkpoint(block, x, causal_mask, key_padding_mask)
            else:
                x = block(x, causal_mask, key_padding_mask)

        x      = self.norm_out(x)
        logits = self.head(x)
        return logits

    def loss(
        self,
        input_ids: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        logits = self.forward(input_ids[:, :-1], key_padding_mask)
        targets = input_ids[:, 1:].contiguous()

        loss = F.cross_entropy(
            logits.reshape(-1, self.cfg["vocab_size"]),
            targets.reshape(-1),
            ignore_index=self.cfg["pad_token_id"],
        )
        return loss

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# Checkpoint helpers
def save_model(model: MusicTransformer, path: str | Path):
    torch.save({"config": model.cfg, "state_dict": model.state_dict()}, path)
    print(f"Model saved → {path}")

def load_model(path: str | Path, device: str = "cpu") -> MusicTransformer:
    ckpt  = torch.load(path, map_location=device)
    state = ckpt.get("model_state") or ckpt.get("state_dict")
    cfg   = ckpt.get("config") or ckpt.get("cfg")
    model = MusicTransformer(config=cfg)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"Model loaded: {path}  ({model.num_parameters():,} params)")
    return model
