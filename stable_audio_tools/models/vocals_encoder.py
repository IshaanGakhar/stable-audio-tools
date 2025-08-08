import torch
from torch import nn
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import typing as tp

class VocalsEncoder(nn.Module):
    def __init__(
        self,
        encoder_model_name: str = "facebook/wav2vec2-base",
        latent_dim: int = 512,
        seq_len: int = 256,
    ):
        super().__init__()
        self.processor = Wav2Vec2Processor.from_pretrained(encoder_model_name)
        self.encoder = Wav2Vec2Model.from_pretrained(encoder_model_name)
        self.encoder.eval()
        self.proj = nn.Linear(self.encoder.config.hidden_size, latent_dim)
        self.seq_len = seq_len
        self.latent_dim = latent_dim

    def encode(self, vocals: torch.Tensor) -> torch.Tensor:
        """
        vocals: (batch, audio_samples)
        Returns: (batch, seq_len, latent_dim)
        """
        # If vocals is not batched, add batch dim
        if vocals.dim() == 1:
            vocals = vocals.unsqueeze(0)
        # Wav2Vec2 expects (batch, audio_samples)
        inputs = self.processor(vocals, sampling_rate=16000, return_tensors="pt", padding=True)
        outputs = self.encoder(inputs.input_values)
        hidden = outputs.last_hidden_state  # (batch, seq, hidden_size)
        # Truncate or pad to seq_len
        if hidden.size(1) > self.seq_len:
            hidden = hidden[:, :self.seq_len, :]
        elif hidden.size(1) < self.seq_len:
            pad = torch.zeros(hidden.size(0), self.seq_len - hidden.size(1), hidden.size(2), device=hidden.device)
            hidden = torch.cat([hidden, pad], dim=1)
        latent = self.proj(hidden)  # (batch, seq_len, latent_dim)
        return latent