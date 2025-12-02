import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class MaskedSelfAttention(nn.Module):
    """
    Masked Self-Attention mechanism for decoder-only transformer.
    Uses causal masking to prevent attending to future positions.
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        Initialize masked self-attention.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of masked self-attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask tensor of shape (batch_size, seq_len, seq_len)
        
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.size()
        
        # Linear projections
        Q = self.w_q(x)  # (batch_size, seq_len, d_model)
        K = self.w_k(x)  # (batch_size, seq_len, d_model)
        V = self.w_v(x)  # (batch_size, seq_len, d_model)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        # Shapes: (batch_size, n_heads, seq_len, d_k)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # scores shape: (batch_size, n_heads, seq_len, seq_len)

        # Always create causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        causal_mask = causal_mask.to(x.device)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

        # Combine with padding mask if provided
        if mask is not None:
            # mask shape: (batch_size, 1, 1, seq_len) from padding
            # Combine with causal mask
            mask = causal_mask | mask
        else:
            mask = causal_mask

        # Apply mask (use large negative value for masked positions)
        scores = scores.masked_fill(mask, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        # context shape: (batch_size, n_heads, seq_len, d_k)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # Final linear projection
        output = self.w_o(context)
        
        return output


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Initialize feed-forward network.
        
        Args:
            d_model: Model dimension
            d_ff: Hidden dimension of feed-forward layer
            dropout: Dropout probability
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of feed-forward network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerDecoderBlock(nn.Module):
    """
    Single transformer decoder block with masked self-attention.
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Initialize transformer decoder block.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            dropout: Dropout probability
        """
        super().__init__()
        self.self_attention = MaskedSelfAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of transformer decoder block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
        
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Self-attention with residual connection and layer norm
        attn_output = self.self_attention(self.norm1(x), mask)
        x = x + self.dropout(attn_output)
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_output)
        
        return x


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
        """
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        
        Returns:
            Input with positional encoding added
        """
        x = x + self.pe[:, :x.size(1)]
        return x


class DecoderOnlyTransformer(nn.Module):
    """
    Decoder-only transformer for language modeling.
    """
    
    def __init__(self, vocab_size: int, d_model: int = 256, n_heads: int = 8, 
                 n_layers: int = 6, d_ff: int = 1024, max_len: int = 30, 
                 dropout: float = 0.1, pad_idx: int = 0):
        """
        Initialize decoder-only transformer.
        
        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            d_ff: Feed-forward hidden dimension
            max_len: Maximum sequence length
            dropout: Dropout probability
            pad_idx: Padding token index
        """
        super().__init__()
        
        self.d_model = d_model
        self.pad_idx = pad_idx
        
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerDecoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].fill_(0)
    
    def create_padding_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        Create padding mask for attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len)
        
        Returns:
            Padding mask of shape (batch_size, 1, 1, seq_len)
        """
        padding_mask = (x == self.pad_idx).unsqueeze(1).unsqueeze(2)
        return padding_mask
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of decoder-only transformer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len)
        
        Returns:
            Logits tensor of shape (batch_size, seq_len, vocab_size)
        """
                
        # Create padding mask
        padding_mask = self.create_padding_mask(x)
        
        # Token embeddings
        x = self.token_embedding(x) * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, padding_mask)
        
        # Final layer norm
        x = self.norm(x)
        
        # Project to vocabulary
        logits = self.output_projection(x)
        
        return logits
    
    def generate(self, max_length: int = 30,
                 temperature: float = 1.0, top_k: Optional[int] = None) -> torch.Tensor:
        """
        Generate sequence autoregressively starting from <start> token.

        Args:
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: If specified, sample from top k tokens

        Returns:
            Generated sequence tensor
        """
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device
            # Use <start> token (typically token index 1)
            start_token = 1
            # Use <end> token (typically token index 2)
            end_token = 2

            generated = torch.tensor([start_token], device=device).unsqueeze(0)

            for _ in range(max_length - 1):
                # Get model predictions
                logits = self.forward(generated)

                # Get logits for next token
                next_token_logits = logits[0, -1, :] / temperature

                # Apply top-k filtering if specified
                if top_k is not None:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, -float('inf'))
                    next_token_logits[top_k_indices] = top_k_logits

                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)

                # Append to generated sequence
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

                # Stop if end token is reached
                if next_token.item() == end_token:
                    break

            return generated.squeeze(0)


def create_model(vocab_size: int, d_model: int = 256, n_heads: int = 8,
                 n_layers: int = 6, d_ff: int = 1024, max_len: int = 30,
                 dropout: float = 0.1, pad_idx: int = 0, learning_rate: float = 1e-4,
                 weight_decay: float = 0.01, num_epochs: int = 100):
    """
    Create transformer model with criterion, optimizer, and scheduler.

    Args:
        vocab_size: Size of vocabulary
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        d_ff: Feed-forward hidden dimension
        max_len: Maximum sequence length
        dropout: Dropout probability
        pad_idx: Padding token index
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for AdamW optimizer
        num_epochs: Number of training epochs for scheduler

    Returns:
        Tuple of (model, criterion, optimizer, scheduler)
    """
    # Create model
    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_len=max_len,
        dropout=dropout,
        pad_idx=pad_idx
    )

    # Create loss criterion
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # Create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,  # Use num_epochs for T_max
        eta_min=learning_rate * 0.01
    )

    return model, criterion, optimizer, scheduler