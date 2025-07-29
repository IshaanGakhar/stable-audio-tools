# stable_audio_tools/models/autoencoders/lyrics_autoencoder.py
import torch
from torch import nn
from torch.nn import functional as F
import typing as tp

# tokenizer, embedder, encoder, VAE

class LyricsAutoencoder(nn.Module):
    def __init__(self,
                 input_dim: int,       # e.g., vocabulary size for one-hot, or embedding dim for pre-trained
                 hidden_dims: tp.List[int],
                 latent_dim: int,
                 seq_len: int):        # Expected input sequence length of lyrics
        super().__init__()
        self.latent_dim = latent_dim
        self.seq_len = seq_len

        # Example: Simple MLP encoder (you'll likely want something more sophisticated for lyrics)
        # If input_dim is a vocab size, you might need an embedding layer first
        # For simplicity, let's assume input_dim is the embedding size per token, or a flattened input.

        encoder_layers = []
        current_dim = input_dim * seq_len # Flatten input for simplicity here, or adapt for sequence processing
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(current_dim, h_dim))
            encoder_layers.append(nn.GELU()) # Or ReLU, etc.
            current_dim = h_dim
        encoder_layers.append(nn.Linear(current_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # You'd also need a decoder if you plan to train this autoencoder
        # self.decoder = nn.Sequential(...)

    def encode(self, lyrics_data: torch.Tensor) -> torch.Tensor:
        # lyrics_data shape: (batch_size, seq_len, input_dim)
        # You might need to flatten or adapt based on your encoder
        # For this example, assuming input_dim is 1 (token ID) and we embed it first
        # Or if input_dim is already an embedding
        
        # Example: Simple flattening for the MLP
        batch_size, seq_len, input_dim = lyrics_data.shape
        flat_lyrics = lyrics_data.view(batch_size, -1) # (batch_size, seq_len * input_dim)
        
        latent = self.encoder(flat_lyrics)
        return latent

    # decode method if it's a VAE or traditional autoencoder

    # For the DiT integration, we primarily care about the 'encode' method