"""Gradient-checkpointed re-implementation of TryOnModel.forward.

Vendor code stays untouched. Identical math — only the mmdit blocks are
wrapped in torch.utils.checkpoint to trade ~35% compute for a large activation
memory cut (11 GB uncheckpointed → fits in 8 GB).
"""
import einops
import torch
from torch.utils.checkpoint import checkpoint

from fashn_vton.tryon_mmdit import apply_conditional_dropout, prepare
from fashn_vton.utils.tensor import unpack_images

rearrange = einops.rearrange


def forward_ckpt(model, x, times, ca_images, garment_images, person_poses,
                 garment_poses, mask=None, garment_categories=None):
    """Mirror of TryOnModel.forward with checkpointed blocks."""
    if mask is None:
        mask = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)

    ca_images = apply_conditional_dropout(ca_images, mask)
    person_poses = apply_conditional_dropout(person_poses, mask)
    x = torch.cat([x, ca_images, person_poses], dim=1)
    x = model.x_embedder(x)
    x, x_ids = prepare(x)

    garment_poses = apply_conditional_dropout(garment_poses, mask)
    garment_images = apply_conditional_dropout(garment_images, mask)
    garment_images = torch.cat([garment_images, garment_poses], dim=1)
    garment_images = model.garment_embedder(garment_images)
    garment_images, garment_ids = prepare(garment_images)

    t = model.t_embedder(times)
    if model.y_embedder is not None and garment_categories is not None:
        y = apply_conditional_dropout(garment_categories, mask)
        t = t + model.y_embedder(y)

    img, txt, vec = x, garment_images, t
    x_pe = model.pe_embedder(x_ids)
    g_pe = model.pe_embedder(garment_ids)

    if model.use_patch_mixer:
        for block in model.x_patch_mixer:
            img = checkpoint(block, img, vec=vec, pe=x_pe, use_reentrant=False)

    pe = torch.cat([x_pe, g_pe], dim=2)

    for block in model.double_blocks:
        img, txt = checkpoint(block, img=img, txt=txt, vec=vec, pe=pe, use_reentrant=False)

    img = torch.cat((txt, img), 1)
    for block in model.single_blocks:
        img = checkpoint(block, img, vec=vec, pe=pe, use_reentrant=False)
    img = img[:, txt.shape[1]:, ...]

    x = model.final_layer(img, vec)
    x = rearrange(x, "b (h w) c -> b c h w",
                  h=model.input_shape[0] // model.patch_size,
                  w=model.input_shape[1] // model.patch_size)
    return {"x": unpack_images(x, model.patch_size)}
