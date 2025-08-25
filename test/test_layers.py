import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
print(parent)
sys.path.append(parent)

import torch
from src.detr.layers import (
    get_positional_encoding,
    get_backbone,
    get_multihead_attention,
    PositionalEncodingSine,
    EncoderLayer,
    Transformer,
)

def test_positional_encoding_shape_and_type():
    """
    Test the shape and type of the positional encoding output.
    """
    H = 64
    W = 128
    d_model = 256
    pe = get_positional_encoding(d_model)
    x = torch.zeros(3, d_model, H, W)
    out = pe(x)
    assert out.shape == (3, d_model, H, W)
    assert isinstance(out, torch.Tensor)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()

def test_backbone_output_shape():
    """
    Test the output shape of the backbone model.
    """
    # Assuming get_backbone returns a callable backbone model
    # that outputs a tensor of shape (batch_size, channels, height, width)
    backbone = get_backbone('resnet50')
    x = torch.randn(2, 3, 224, 224)
    out = backbone(x)
    assert isinstance(out, torch.Tensor)
    assert out.shape[0] == 2  # batch size
    assert out.shape[1] in [2048]  # channels, depending on dilation
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()

def test_multihead_attention_forward():
    """
    Test the forward pass of the multihead attention module.
    """
    d_model = 128
    nhead = 8
    mha = get_multihead_attention(d_model, nhead)
    x = torch.randn(2, 10, d_model)
    out, _ = mha(x, x, x)
    assert out.shape == (2, 10, d_model)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()

def test_encoder_layer_forward():
    """
    Test the forward pass of the encoder layer.
    """
    d_model = 128
    nhead = 8
    dim_feedforward = 256
    layer = EncoderLayer(d_model, nhead=nhead, dim_feedforward=dim_feedforward)
    x = torch.randn(2, 10, d_model)
    out = layer(x)
    assert out.shape == (2, 10, d_model)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()

def test_transformer_forward():
    """
    Test the forward pass of the transformer module.
    """
    d_model = 128
    nhead = 8
    num_encoder_layers = 2
    num_decoder_layers = 2
    dim_feedforward = 256
    transformer = Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward)
    src = torch.randn(2, 10, d_model)
    pos_encoding = torch.zeros(1, 10, d_model)
    query_embedding = torch.zeros(2, 10, d_model)
    out = transformer(src, pos_encoding, query_embedding)
    assert isinstance(out, torch.Tensor) or (isinstance(out, tuple) and all(isinstance(t, torch.Tensor) for t in out))