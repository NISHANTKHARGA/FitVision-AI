"""Pre-flight smoke test: load TryOnModel + LoRA on GPU, one forward+backward on random tensors.

Run:  python scripts/smoke_train.py
Purpose: catch API/OOM/dtype bugs before the real dataset is involved.
Does NOT prove quality — only that the training step runs.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ai" / "fashn-vton-1.5" / "src"))

from fashn_vton.tryon_mmdit import TryOnModel  # noqa: E402
from fashn_vton.utils.checkpoint import load_checkpoint  # noqa: E402
from training.ckpt_forward import forward_ckpt  # noqa: E402
from training.lora import inject_lora, freeze_base, lora_parameters  # noqa: E402

device = "cuda" if torch.cuda.is_available() else "cpu"
weights = Path(__file__).resolve().parents[2] / "ai" / "fashn-vton-1.5" / "weights" / "model.safetensors"
print(f"Loading TryOnModel on {device} (bf16)…")
model = TryOnModel()
state = load_checkpoint(str(weights), device=device)
model.load_state_dict(state, strict=False)
model.to(device=device, dtype=torch.bfloat16)
model.train()

replaced = inject_lora(model, rank=16, alpha=32,
                       target_filters=("qkv", "proj", "mlp.0", "mlp.2", "linear1", "linear2"))
freeze_base(model)
print(f"LoRA injected into {len(replaced)} linears; trainable {sum(p.numel() for p in lora_parameters(model))/1e6:.1f}M")

B, C, H, W = 1, 3, 864, 576
xx = torch.randn(B, C, H, W, device=device, dtype=torch.bfloat16)
ca = torch.randn(B, C, H, W, device=device, dtype=torch.bfloat16)
gi = torch.randn(B, C, H, W, device=device, dtype=torch.bfloat16)
pp = torch.randn(B, 1, H, W, device=device, dtype=torch.bfloat16)
gp = torch.randn(B, 1, H, W, device=device, dtype=torch.bfloat16)
cat = torch.ones(B, dtype=torch.long, device=device)
t = torch.rand(B, device=device, dtype=torch.bfloat16)

torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
pred = forward_ckpt(model, xx, t, ca_images=ca, garment_images=gi, person_poses=pp, garment_poses=gp,
                    garment_categories=cat)["x"]
loss = torch.nn.functional.mse_loss(pred.float(), torch.randn_like(pred.float()))
loss.backward()
print(f"forward+backward OK · loss {loss.item():.5f}")
if device == "cuda":
    print(f"peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
print("SMOKE OK")
