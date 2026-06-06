import torch
import torch.nn as nn
import torch.nn.functional as F

class StraightThroughEstimator(torch.autograd.Function):
    """
    Allows gradients to pass through the discrete rounding function unchanged.
    Essential for updating continuous latent master weights.
    """
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def quantize_weights_158(W):
    """
    Quantizes weights into ternary values: -1, 0, or 1.
    Scale factor beta represents the average absolute value matrix-wide.
    """
    beta = torch.mean(torch.abs(W)) + 1e-9
    scaled_W = W / beta
    clipped_W = torch.clamp(scaled_W, -1.0, 1.0)
    ternary_W = StraightThroughEstimator.apply(clipped_W)
    return ternary_W, beta

def quantize_activations_8bit(x):
    """
    Quantizes activations to 8-bit integers to optimize matrix operations.
    """
    gamma = torch.max(torch.abs(x)) + 1e-9
    scaled_x = x * (127.0 / gamma)
    clipped_x = torch.clamp(scaled_x, -128.0, 127.0)
    quantized_x = StraightThroughEstimator.apply(clipped_x)
    return quantized_x, gamma

class BitLinear(nn.Module):
    """
    Replaces standard nn.Linear. Holds high-precision master weights,
    but executes using ternary weights during the forward pass.
    """
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Continuous master weights (updated by optimizer)
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        # 1. Quantize weights and activations
        ternary_W, beta = quantize_weights_158(self.weight)
        quantized_x, gamma = quantize_activations_8bit(x)
        
        # 2. Perform low-precision linear operation
        output = F.linear(quantized_x, ternary_W)
        
        # 3. De-scale output back to floating point range
        output = output * (beta * gamma / 127.0)
        
        if self.bias is not None:
            output += self.bias
            
        return output

class BitNetTransformerBlock(nn.Module):
    """
    A single 1.58-bit Transformer Layer utilizing custom BitLinear layers.
    """
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        
        # Attention Projections via BitLinear
        self.q_proj = BitLinear(d_model, d_model)
        self.k_proj = BitLinear(d_model, d_model)
        self.v_proj = BitLinear(d_model, d_model)
        self.out_proj = BitLinear(d_model, d_model)
        
        self.ffn_norm = nn.LayerNorm(d_model)
        
        # SwiGLU styled FFN projections using BitLinear
        self.gate_proj = BitLinear(d_model, d_ff)
        self.down_proj = BitLinear(d_ff, d_model)
        self.up_proj = BitLinear(d_model, d_ff)
        
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

    def self_attention(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Memory-efficient scaled dot product attention
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

    def forward(self, x):
        # Residual connection over Attention Layer
        x = x + self.self_attention(self.attn_norm(x))
        
        # Residual connection over SwiGLU FFN Layer
        norm_x = self.ffn_norm(x)
        ffn_out = self.down_proj(F.silu(self.gate_proj(norm_x)) * self.up_proj(norm_x))
        x = x + ffn_out
        return x

class BitNetLM(nn.Module):
    """
    The full language model architecture, stacking the transformer blocks
    and handling token embeddings and output projections.
    """
    def __init__(self, vocab_size=50257, d_model=1024, n_layers=16, n_heads=16, d_ff=2730, max_seq_len=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        
        self.layers = nn.ModuleList([
            BitNetTransformerBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie embeddings and final weights
        self.lm_head.weight = self.token_emb.weight

    def forward(self, idx, use_checkpointing=False):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        
        x = self.token_emb(idx) + self.pos_emb(pos)
        
        for layer in self.layers:
            if use_checkpointing and self.training:
                from torch.utils.checkpoint import checkpoint
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
            
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits
