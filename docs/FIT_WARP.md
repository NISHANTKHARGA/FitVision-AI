# Fit Preference Simulation — Method & Honesty Note

## What the try-on model can and cannot do

FASHN VTON v1.5 (used by FitVision AI) accepts: person image, garment image,
`category` (`tops` | `bottoms` | `one-pieces`), `garment_photo_type`, `num_timesteps`,
`guidance_scale`, `seed`, `segmentation_free`.

**It has no "fit" control.** Neither do any other open local try-on models we
evaluated (CatVTON, IDM-VTON — both also require non-commercial licensing,
see `docs/MODEL_SELECTION.md`).

## What FitVision AI does instead

Slim / Regular / Oversized previews are produced **after** the diffusion render by a
documented, deterministic image-processing step:

1. **Anatomical anchoring.** MediaPipe Pose landmarks locate the torso axis
   (`cx = mean(left/right shoulder x)`), shoulder height, hip height and image bounds.
2. **Garment band selection.**
   - *Tops:* from `shoulder_y − 0.28·torso_h` to `hip_y + 0.45·torso_h`
   - *Bottoms:* from `hip_y − 0.08·torso_h` to `ankle_y (+ margin)` or the image bottom
3. **Smooth scale profile.** For each row `y` in the band, a sine bump
   `w(y) = sin(π·t)^0.85` (t = normalized position in the band) modulates a horizontal
   scale around `cx`:
   - slim → mid-band scale ≈ 0.92
   - regular → 1.0 (identity — the untouched model output)
   - oversized → ≈ 1.09, plus a ~3% vertical stretch about the band centre
   Shoulders (band edges) are least affected; chest/waist (mid-band) most. This is a
   **non-uniform, anatomy-anchored warp** — not a uniform resize.
4. **Resampling.** `cv2.remap` with bilinear interpolation and replicate borders.
   No pixel is invented or redrawn; the garment's colours, prints and logos are
   resampled, not regenerated, so design fidelity is preserved by construction.
5. **Feather-blending.** Band edges are alpha-feathered over ~12% of the band height
   so the warp merges seamlessly with the untouched head, arms, legs and background.

## What this means — and what it does not

| Claim | Status |
|---|---|
| "These pixels came from the AI try-on output, geometrically resampled" | ✅ True |
| "Oversized reads as visually wider silhouette" | ✅ True (that's the point of the view) |
| "This is how the garment will drape / hang / fit your body" | ❌ Not claimed — fabric physics is out of scope |
| "This is a physical fit prediction" | ❌ Never claimed |

Fit previews are explicitly labelled **"visual approximation"** in the UI, and the
verdict-style **measurement-based fit analysis** (separate feature,
`fit_analysis.py`) is the honest, quantitative fit signal — driven by your body and
garment measurements, never by the rendered image.
