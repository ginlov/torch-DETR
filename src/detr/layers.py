import torch

from torch.nn import functional as F
from torchvision.ops import FrozenBatchNorm2d

def get_backbone(backbone_name) -> torch.nn.Module:
    """
    Returns a backbone model based on the provided name.
    
    Args:
        backbone_name (str): The name of the backbone model to retrieve.
        
    Returns:
        torch.nn.Module: The backbone model.
    """
    # Load a pre-trained ResNet model
    # Remove the last two layers
    # Use frozen batch normalization and replace stride with dilation as described in the DETR
    if backbone_name == 'resnet18':
        from torchvision.models import resnet18, ResNet18_Weights
        model = resnet18(weights=ResNet18_Weights.DEFAULT, norm_layer=FrozenBatchNorm2d, replace_stride_with_dilation=[False, False, True])
        backbone = torch.nn.Sequential(*list(model.children())[:-2])
        backbone.out_channels = 512
        return backbone
    elif backbone_name == 'resnet50':
        from torchvision.models import resnet50, ResNet50_Weights
        model = resnet50(weights=ResNet50_Weights.DEFAULT, norm_layer=FrozenBatchNorm2d, replace_stride_with_dilation=[False, False, True])
        backbone = torch.nn.Sequential(*list(model.children())[:-2])
        backbone.out_channels = 2048
        return backbone
    elif backbone_name == 'resnet101':
        from torchvision.models import resnet101, ResNet101_Weights
        model = resnet101(weights=ResNet101_Weights.DEFAULT, norm_layer=FrozenBatchNorm2d, replace_stride_with_dilation=[False, False, True])
        backbone = torch.nn.Sequential(*list(model.children())[:-2])
        backbone.out_channels = 2048
        return backbone
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

def get_positional_encoding(d_model) -> torch.nn.Module:
    """
    Returns a positional encoding module.
    
    Returns:
        torch.nn.Module: The positional encoding module.
    """
    # Placeholder for actual implementation
    # This function should return a specific positional encoding module
    return PositionalEncodingSine(d_model=d_model)

def get_multihead_attention(d_model, nhead):
    """
    Returns a multihead attention module.
    
    Args:
        d_model (int): The number of expected features in the input.
        nhead (int): The number of heads in the multiheadattention models.
        
    Returns:
        torch.nn.Module: The multihead attention module.
    """
    return torch.nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)

class PositionalEncodingSine(torch.nn.Module):
    """
    2D sine-cosine positional encoding for images, as used in DETR.
    Args:
        d_model (int): The number of expected features in the input.
    """
    # TODO: Review the implementation and ensure it matches the DETR paper's description.
    def __init__(self, d_model):
        super(PositionalEncodingSine, self).__init__()
        self.d_model = d_model
        if d_model % 2 != 0:
            raise ValueError("d_model must be even for 2D positional encoding.")

    def forward(self, x):
        """
        Forward pass for positional encoding.
        Args:
            x (torch.Tensor): Input tensor of shape (N, S, H, W)
        Returns:
            torch.Tensor: Positional encoding tensor of the same shape as input x.
        """
        N, S, H, W = x.shape
        device = x.device

        y_position = torch.arange(H, dtype=torch.float32, device=device).unsqueeze(1).repeat(1, W)
        x_position = torch.arange(W, dtype=torch.float32, device=device).unsqueeze(0).repeat(H, 1)

        div_term = torch.exp(torch.arange(0, S // 2, 2, device=device).float() * (-torch.log(torch.tensor(10000.0, device=device)) / (S // 2)))

        pe_y = torch.zeros(S // 2, H, W, device=device)
        pe_x = torch.zeros(S // 2, H, W, device=device)
        for i in range(0, S // 2, 2):
            pe_y[i, :, :] = torch.sin(y_position * div_term[i // 2])
            pe_y[i + 1, :, :] = torch.cos(y_position * div_term[i // 2])
            pe_x[i, :, :] = torch.sin(x_position * div_term[i // 2])
            pe_x[i + 1, :, :] = torch.cos(x_position * div_term[i // 2])
        pe = torch.zeros(S, H, W, device=device)
        pe[:S // 2, :, :] = pe_y
        pe[S // 2:, :, :] = pe_x

        pos = pe.unsqueeze(0).repeat(N, 1, 1, 1)  # (N, S, H, W)
        return pos

class Transformer(torch.nn.Module):
    """
    A transformer module that consists of encoder and decoder layers.
    
    Args:
        d_model (int): The number of expected features in the input.
        nhead (int): The number of heads in the multiheadattention models.
        num_encoder_layers (int): The number of encoder layers.
        num_decoder_layers (int): The number of decoder layers.
        dim_feedforward (int): The dimension of the feedforward network model.
        dropout (float): The dropout value.
    """
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward=2048, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder = torch.nn.ModuleList([
            EncoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_encoder_layers)
        ])
        self.decoder = torch.nn.ModuleList([
            DecoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_decoder_layers)
        ])

    def forward(self, src, pos_encoding, query_embedding): # query embedding is learnt positional encoding as described in the DETR paper
        """
        src: Input tensor of shape (N, d_model, H x W)
        pos_encoding: Positional encoding tensor of shape (N, d_model, H x W)
        query_embedding: Query embedding tensor of shape (num_queries, d_model)
        """
        tgt = torch.zeros_like(query_embedding)  # Initialize target with zeros as described in the DETR paper
        for layer in self.encoder:
            src = layer(src, positional_encoding=pos_encoding)

        for layer in self.decoder:
            tgt = layer(tgt, src, positional_encoding=query_embedding, encoder_positional_encoding=pos_encoding)
        
        return tgt

class DecoderLayer(torch.nn.Module):
    """
    A single decoder layer for the transformer.
    
    Args:
        d_model (int): The number of expected features in the input.
        nhead (int): The number of heads in the multiheadattention models.
        dim_feedforward (int): The dimension of the feedforward network model.
        dropout (float): The dropout value.
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = get_multihead_attention(d_model, nhead)
        self.multihead_attn = get_multihead_attention(d_model, nhead)
        self.conv1 = torch.nn.Conv1d(d_model, dim_feedforward, kernel_size=1)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv1d(dim_feedforward, d_model, kernel_size=1)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.norm3 = torch.nn.LayerNorm(d_model)

    def forward(self, tgt, memory, positional_encoding=None, encoder_positional_encoding=None):
        # tgt: N x T x E
        # memory: N x S x E
        if positional_encoding is not None:
            tgt_with_pos = tgt + positional_encoding
            tgt2 = self.self_attn(tgt_with_pos, tgt_with_pos, tgt)[0]
        else:
            tgt2 = self.self_attn(tgt, tgt, tgt)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm1(tgt)

        if encoder_positional_encoding is not None:
            mem_with_enc_pos = memory + encoder_positional_encoding
        else:
            mem_with_enc_pos = memory

        if positional_encoding is not None:
            tgt_with_pos = tgt + positional_encoding
        else:
            tgt_with_pos = tgt

        # Multihead attention with memory
        tgt2 = self.multihead_attn(tgt_with_pos, mem_with_enc_pos, memory)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm2(tgt)

        # Feedforward with 1x1 conv: (N, T, E) -> (N, E, T) for Conv1d
        tgt_conv = tgt.transpose(1, 2)  # N x E x T
        tgt2 = self.conv2(self.dropout(torch.relu(self.conv1(tgt_conv))))
        tgt2 = tgt2.transpose(1, 2)  # N x T x E
        tgt = tgt + self.dropout(tgt2)
        
        return self.norm3(tgt)

class EncoderLayer(torch.nn.Module):
    """
    A single encoder layer for the transformer.
    
    Args:
        d_model (int): The number of expected features in the input.
        nhead (int): The number of heads in the multiheadattention models.
        dim_feedforward (int): The dimension of the feedforward network model.
        dropout (float): The dropout value.
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = get_multihead_attention(d_model, nhead=nhead)
        self.conv1 = torch.nn.Conv1d(d_model, dim_feedforward, kernel_size=1)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv1d(dim_feedforward, d_model, kernel_size=1)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        
    def forward(self, src, positional_encoding=None):
        """
        src: Input tensor of shape (N, d_model, H x W)
        positional_encoding: Optional positional encoding tensor of shape (N, d_model, H x W)
        """
        # src: N x S x E
        # positional_encoding: N x S x E (optional)
        if positional_encoding is not None:
            src_with_pos = src + positional_encoding
            src2 = self.self_attn(src_with_pos, src_with_pos, src)[0]
        else:
            src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout(src2)
        src = self.norm1(src)
        # Feedforward with 1x1 conv: (N, S, E) -> (N, E, S) for Conv1d
        src_conv = src.transpose(1, 2)  # N x E x S
        src2 = self.conv2(self.dropout(torch.relu(self.conv1(src_conv))))
        src2 = src2.transpose(1, 2)  # N x S x E
        src = src + self.dropout(src2)
        return self.norm2(src)

class FFN(torch.nn.Module):
    """
    A feedforward neural network module.
    
    Args:
        d_model (int): The number of expected features in the input.
        hidden_dim (int): The dimension of the FFN.
        d_out (int): The dimension of the output.
    """
    def __init__(self, d_model, hidden_dim, d_out):
        super(FFN, self).__init__()
        self.linear1 = torch.nn.Linear(d_model, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, d_out)
        
    def forward(self, x):
        # x: N x T x E
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        return x