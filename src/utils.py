from types import SimpleNamespace
from typing import Dict

import torch

# TODO: Implement nested tensor

CONSTANTS = SimpleNamespace(
    NO_OBJECT_LABEL=0, IMAGE_EXTENSIONS=(".jpg", ".jpeg", ".png")
)


def init_weights(module):
    """
    Initialize the weights of the model.

    Args:
        module (torch.nn.Module): The module to initialize.
    """
    xavier_uniform_ = [
        torch.nn.Conv2d,
        torch.nn.Linear,
        torch.nn.ConvTranspose2d,
        torch.nn.Conv1d,
        torch.nn.ConvTranspose1d,
    ]
    if any(isinstance(module, m) for m in xavier_uniform_):
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            torch.nn.init.constant_(module.bias, 0)
    elif isinstance(module, torch.nn.LayerNorm):
        torch.nn.init.constant_(module.bias, 0)
        torch.nn.init.constant_(module.weight, 1.0)
    elif isinstance(module, torch.nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0, std=0.01)


@torch.no_grad
def cal_param_norm(model: torch.nn.Module) -> Dict:
    """Compute parameter L2 norm

    Args:
        model (torch.nn.Module): DETR model

    Returns:
        Dict: norm of each layer in DETR model
    """
    param_norm = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            norm = torch.norm(param, p=2).item()
            param_norm[f"param_norm/{name}"] = norm

    return param_norm


@torch.no_grad
def cal_grad_norm(model: torch.nn.Module) -> Dict:
    """Computer gradient norm

    Args:
        optimizer (torch.optim.Optimizer): optimizer

    Returns:
        Dict: gradient norm dictionary
    """
    grad_norm = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            norm = torch.norm(param.grad, p=2).item()
            grad_norm[f"grad_norm/{name}"] = norm
    return grad_norm

def get_warmup_cosine_schedule(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int
):
    """
    Create a schedule with a learning rate that decreases linearly after a warmup period.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which to schedule the learning rate.
        num_warmup_steps (int): The number of steps for the warmup phase.
        num_training_steps (int): The total number of training steps.

    Rteturns:
        torch.optim.lr_scheduler.LambdaLR: The learning rate scheduler.
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return max(1e-2, float(current_step) / float(max(1, num_warmup_steps)))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(1e-3, 0.5 * (1.0 + torch.cos(torch.pi * torch.tensor(progress))))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
