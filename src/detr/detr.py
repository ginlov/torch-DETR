import torch

from src.detr.layers import Transformer, get_backbone, get_positional_encoding
from src.utils import init_weights


class DETR(torch.nn.Module):
    """
    The main DETR model.

    Args:
        backbone (torch.nn.Module): The backbone model to extract features.
        transformer (torch.nn.Module): The transformer module.
        d_model (int): The dimension of the model (Transformer).
        num_classes (int): The number of object classes.
        num_queries (int): The number of queries for the transformer.
    """

    def __init__(self, backbone, transformer, d_model, num_classes, num_queries):
        super(DETR, self).__init__()
        self.backbone = backbone
        self.conv_project = torch.nn.Conv2d(
            backbone.out_channels, d_model, kernel_size=1
        )
        self.transformer = transformer
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.positional_encoding = get_positional_encoding(d_model)
        self.query_embedding = torch.nn.Embedding(num_queries, d_model)

        # Initialize the classification and bounding box prediction heads
        self.class_embed = torch.nn.Linear(
            d_model, num_classes + 1
        )  # +1 for "no object" class
        self.bbox_embed = torch.nn.Linear(d_model, 4)  # 4 for bounding box coordinates

        # Initialize weights to make training consistent
        self.conv_project.apply(init_weights)
        self.query_embedding.apply(init_weights)
        self.class_embed.apply(init_weights)
        self.bbox_embed.apply(init_weights)

    def forward(self, x, mask: torch.Tensor = None):
        """
        Forward pass of the DETR model.

        Args:
            x (torch.Tensor): Input tensor. (N, 3, H0, W0)
            mask (torch.Tensor): Mask tensor. (N, H0, W0) 1 is ignore, 0 is keep

        Returns:
            tuple: Class predictions and bounding box predictions.
        """
        src = self.backbone(
            x
        )  # N x backbone.out_channels x H0/32 x W0/32 = N x backbond.out_channels x H x W
        if mask is not None:
            mask = (
                torch.nn.functional.interpolate(mask[None].float(), size=src.shape[-2:])
                .to(torch.bool)[0]
                .flatten(1)
            )  # N x (H * W) # interpolate to the size of src
        src = self.conv_project(src)  # N x d_model x H x W

        ## Positional encoding is fixed, so no need to pass mask
        pos_encoding = (
            self.positional_encoding(src).flatten(2).permute(0, 2, 1)
        )  # N x (H*W) x d_model
        src = src.flatten(2).permute(0, 2, 1)  # N x (H*W) x d_model

        ## Feed to transformer
        tgt = self.transformer(
            src, pos_encoding, self.query_embedding.weight.repeat(x.size(0), 1, 1), mask
        )  # N x num_queries x d_model

        class_preds = self.class_embed(tgt)
        bbox_preds = self.bbox_embed(tgt)

        return class_preds, bbox_preds


def build_detr(
    backbone_name,
    num_classes,
    num_queries,
    d_model=256,
    nhead=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=2048,
    dropout=0.1,
):
    """
    Builds the DETR model with the specified backbone and transformer parameters.

    Args:
        backbone_name (str): The name of the backbone model.
        num_classes (int): The number of object classes.
        num_queries (int): The number of queries for the transformer.
        d_model (int): The number of expected features in the input.
        nhead (int): The number of heads in the multiheadattention models.
        num_encoder_layers (int): The number of encoder layers.
        num_decoder_layers (int): The number of decoder layers.
        dim_feedforward (int): The dimension of the feedforward network model.
        dropout (float): The dropout value.

    Returns:
        DETR: The constructed DETR model.
    """
    backbone = get_backbone(backbone_name)
    transformer = Transformer(
        d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout
    )

    return DETR(backbone, transformer, d_model, num_classes, num_queries)
