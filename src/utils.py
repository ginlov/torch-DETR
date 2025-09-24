import torch
from types import SimpleNamespace

#TODO: Implement nested tensor

CONSTANTS = SimpleNamespace(
    NO_OBJECT_LABEL=0,
    IMAGE_EXTENSIONS=('.jpg', '.jpeg', '.png')
)

def pad_targets(targets, num_queries: int):
    """
    Convert list of dict targets into padded tensors for DETR-style models.

    Args:
        targets (list): Each element is {"boxes": Tensor[num_obj, 4], "labels": Tensor[num_obj]}
        num_queries (int): Number of queries (N) for padding/truncation.

    Returns:
        boxes_batch: Tensor [batch_size, num_queries, 4]
        labels_batch: Tensor [batch_size, num_queries]
    """
    batch_size = len(targets)

    # Initialize with padding
    boxes_batch = torch.zeros((batch_size, num_queries, 4), dtype=torch.float32)
    labels_batch = torch.full((batch_size, num_queries), CONSTANTS.NO_OBJECT_LABEL, dtype=torch.long)  # -1 = "no object"

    for i, target in enumerate(targets):
        boxes = target["boxes"]
        labels = target["labels"]

        n = min(len(boxes), num_queries)  # truncate if too many objects
        boxes_batch[i, :n] = boxes[:n]
        labels_batch[i, :n] = labels[:n]

    return labels_batch, boxes_batch

def init_weights(module):
    """
    Initialize the weights of the model.
    
    Args:
        module (torch.nn.Module): The module to initialize.
    """
    xavier_uniform_ = [torch.nn.Conv2d, torch.nn.Linear, torch.nn.ConvTranspose2d, torch.nn.Conv1d, torch.nn.ConvTranspose1d]
    if any(isinstance(module, m) for m in xavier_uniform_):
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            torch.nn.init.constant_(module.bias, 0)
    elif isinstance(module, torch.nn.LayerNorm):
        torch.nn.init.constant_(module.bias, 0)
        torch.nn.init.constant_(module.weight, 1.0)
    elif isinstance(module, torch.nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0, std=0.01)