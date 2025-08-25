import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import torch
from src.detr.detr import build_detr

def test_detr_forward_shapes():
    """
    Test the forward pass of the DETR model to ensure output shapes are correct.
    """
    model = build_detr('resnet50', num_classes=91, num_queries=100)
    dummy_input = torch.randn(2, 3, 224, 224)
    class_preds, bbox_preds = model(dummy_input)
    assert class_preds.shape == (2, 100, 91)
    assert bbox_preds.shape == (2, 100, 4)

def test_detr_no_nan_or_inf():
    """
    Test that the DETR model outputs do not contain NaN or Inf values.
    """
    model = build_detr('resnet50', num_classes=91, num_queries=100)
    dummy_input = torch.randn(2, 3, 224, 224)
    class_preds, bbox_preds = model(dummy_input)
    assert not torch.isnan(class_preds).any(), "NaN detected in class_preds"
    assert not torch.isnan(bbox_preds).any(), "NaN detected in bbox_preds"
    assert not torch.isinf(class_preds).any(), "Inf detected in class_preds"
    assert not torch.isinf(bbox_preds).any(), "Inf detected in bbox_preds"

def test_detr_gradient_flow():
    """
    Test that gradients flow through the DETR model during backpropagation.
    """
    model = build_detr('resnet50', num_classes=91, num_queries=100)
    dummy_input = torch.randn(2, 3, 224, 224, requires_grad=True)
    class_preds, bbox_preds = model(dummy_input)
    loss = class_preds.sum() + bbox_preds.sum()
    loss.backward()
    no_grad_params = []
    name_of_params_not_receiving_grad = ['transformer.decoder.0.self_attn.in_proj_weight', 'transformer.decoder.0.self_attn.out_proj.weight', 'transformer.decoder.0.norm1.weight']
    for name, param in model.named_parameters():
        if param.requires_grad and (param.grad is None or not torch.any(param.grad != 0)) and name not in name_of_params_not_receiving_grad:
            no_grad_params.append(name)
    if no_grad_params:
        print("Parameters with no gradient:", no_grad_params)
    assert len(no_grad_params) == 0, f"Some parameters did not receive gradients: {no_grad_params}"

def test_detr_module_integration():
    """
    Test that the DETR model can be built and its submodules are correctly initialized.
    """
    model = build_detr('resnet50', num_classes=91, num_queries=100)
    # Check submodules exist and are callable
    assert hasattr(model, "backbone")
    assert hasattr(model, "transformer")
    assert hasattr(model, "class_embed")
    assert hasattr(model, "bbox_embed")
    # Check forward pass works
    dummy_input = torch.randn(1, 3, 224, 224)
    class_preds, bbox_preds = model(dummy_input)
    assert class_preds is not None and bbox_preds is not None