"""Minimal LoRA injection for TryOnModel's attention/MLP linear layers.

Keeps the base FASHN checkpoint frozen; trains only rank-r update matrices.
Merge back with `save_merged_checkpoint()` for a drop-in replacement checkpoint.
"""
import math
from typing import Dict, List

import torch
from torch import nn


class LoRALinear(nn.Linear):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__(base.in_features, base.out_features,
                         bias=base.bias is not None,
                         device=base.weight.device, dtype=base.weight.dtype)
        # share the frozen base weights (no copy of the 640 MB per layer)
        self.weight = base.weight
        self.bias = base.bias
        self.rank = rank
        self.scaling = alpha / rank
        # LoRA params in fp32 for optimizer stability
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32, device=base.weight.device))
        self.lora_B = nn.Parameter(torch.empty(base.out_features, rank, dtype=torch.float32, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        base_out = torch.nn.functional.linear(x, self.weight, self.bias)
        lora_dtype = self.lora_A.dtype
        lora_out = (x.to(lora_dtype) @ self.lora_A.T) @ self.lora_B.T
        return base_out + (lora_out * self.scaling).to(base_out.dtype)

    def merged_weight(self) -> torch.Tensor:
        delta = (self.lora_B @ self.lora_A) * self.scaling
        return (self.weight.to(torch.float32) + delta).to(self.weight.dtype)


def inject_lora(model: nn.Module, rank: int = 16, alpha: float = 32.0,
                target_filters: tuple = ("qkv", "proj", "fc1", "fc2")) -> Dict[str, LoRALinear]:
    """Replace attention/MLP linears in mmdit blocks with LoRALinear (freeze the rest)."""
    replaced: Dict[str, LoRALinear] = {}

    def _replace(parent: nn.Module, prefix: str):
        for name, child in list(parent.named_children()):
            path = f"{prefix}{name}"
            if isinstance(child, nn.Linear) and any(t in name for t in target_filters):
                if any(b in path for b in ("double_blocks", "single_blocks", "x_patch_mixer")):
                    new = LoRALinear(child, rank, alpha)
                    # replace the submodule while preserving base references
                    parent._modules[name] = new
                    replaced[path] = new
            else:
                _replace(child, path + ".")

    _replace(model, "")
    return replaced


def freeze_base(model: nn.Module):
    for name, param in model.named_parameters():
        param.requires_grad = "lora_" in name


def lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    return [p for n, p in model.named_parameters() if "lora_" in n]


@torch.no_grad()
def merge_lora_(model: nn.Module):
    for parent in model.modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, LoRALinear):
                new_lin = nn.Linear(child.in_features, child.out_features,
                                    bias=child.bias is not None,
                                    device=child.weight.device,
                                    dtype=child.weight.dtype)
                new_lin.weight.copy_(child.merged_weight())
                if child.bias is not None:
                    new_lin.bias.copy_(child.bias)
                parent._modules[name] = new_lin


def save_merged_checkpoint(model: nn.Module, out_path: str):
    """Merge LoRA into the base weights and save a drop-in FASHN-format checkpoint."""
    import os
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    merge_lora_(model)
    from safetensors.torch import save_file
    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    save_file(state, out_path, metadata={"format": "pt"})
