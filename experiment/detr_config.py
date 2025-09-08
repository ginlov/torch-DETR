from dataclasses import dataclass

@dataclass
class DETRConfig:
    d_model: int = 256
    nhead: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dim_feedforward: int =2048
    dropout: float = 0.1