import torch
from torch import nn
import typing as tp

from transformers import T5EncoderModel, T5Tokenizer
from encodec.utils import EncodecTokenizer

class LyricsAutoencoder(nn.Module):
    def __init__(
        self,
        encoder_model_name: str = "t5-base", # google/flan-t5-base also works
        latent_dim: int = 1024, # same as MusicGen-small
        seq_len: int = 256, # same as MusicGen
    ):
        super().__init__()
        # T5 setup
        self.t5_tokenizer = T5Tokenizer.from_pretrained(encoder_model_name)
        self.t5_encoder = T5EncoderModel.from_pretrained(encoder_model_name)
        self.t5_encoder.eval()  # Usually frozen for inference

        # Project T5 output to latent_dim
        self.t5_proj = nn.Linear(self.t5_encoder.config.d_model, latent_dim)

        self.seq_len = seq_len
        self.latent_dim = latent_dim

    def encode(self, lyrics: tp.List[str], audio: torch.Tensor = None) -> torch.Tensor:
        """
        lyrics: List of lyric strings (batch)
        Returns: (batch, seq_len, latent_dim)
        Example:
            lyrics = [
                "We're no strangers to love, you know the rules and so do I.",
                "Never gonna give you up, never gonna let you down."
            ]
        """
        # T5 encoding
        t5_inputs = self.t5_tokenizer(
            lyrics, 
            return_tensors="pt", 
            padding="max_length", 
            max_length=self.seq_len, 
            truncation=True
            )
        
        t5_outputs = self.t5_encoder(
            input_ids=t5_inputs.input_ids, 
            attention_mask=t5_inputs.attention_mask
            )
        
        t5_latent = self.t5_proj(t5_outputs.last_hidden_state
                                 )  # (batch, seq_len, latent_dim)
        
        return t5_latent