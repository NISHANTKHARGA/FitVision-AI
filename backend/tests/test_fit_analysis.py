"""Unit tests for the pure (non-ML) backend logic."""
import numpy as np

from app.services.fit_analysis import analyze_fit
from app.services.fit_warp import fit_warp
from app.services.recommendations import recommend


def test_fit_analysis_tight_chest():
    r = analyze_fit({"chest_cm": 110, "height_cm": 175, "shoulder_cm": 46},
                    {"chest_width_cm": 48}, "tops", "slim")
    chest = next(v for v in r["verdicts"] if v["area"] == "Chest")
    assert chest["status"] == "tight"
    assert "rules_used" in r and r["rules_used"]


def test_fit_analysis_ideal_regular():
    r = analyze_fit({"chest_cm": 96}, {"chest_width_cm": 55}, "tops", "regular")
    chest = next(v for v in r["verdicts"] if v["area"] == "Chest")
    assert chest["status"] == "ideal"  # ease 14 cm


def test_fit_analysis_no_data():
    r = analyze_fit({}, {}, "tops", "regular")
    assert r["verdicts"] == []
    assert "Not enough measurements" in r["overall"]


def test_fit_analysis_bottoms():
    r = analyze_fit({"waist_cm": 82, "hip_cm": 98},
                    {"waist_width_cm": 47, "hip_width_cm": 52},
                    "bottoms", "regular")
    areas = {v["area"] for v in r["verdicts"]}
    assert "Waist" in areas and "Hip" in areas


def test_fit_warp_shapes_and_identity():
    img = np.random.randint(0, 255, (400, 300, 3), dtype=np.uint8)
    kp = {"center_x": 150.0, "shoulder_y": 80.0, "hip_y": 220.0,
          "shoulder_width_px": 90.0, "hip_width_px": 100.0,
          "ankle_y": 380.0, "image_size": (300, 400)}
    slim = fit_warp(img, kp, "tops", "slim")
    over = fit_warp(img, kp, "tops", "oversized")
    reg = fit_warp(img, kp, "tops", "regular")
    assert slim.shape == img.shape and over.shape == img.shape
    assert np.array_equal(reg, img)          # regular = identity
    assert not np.array_equal(slim, img)     # slim actually changes pixels
    assert not np.array_equal(over, img)


def test_fit_warp_without_pose_is_safe():
    img = np.random.randint(0, 255, (200, 100, 3), dtype=np.uint8)
    out = fit_warp(img, None, "tops", "oversized")
    assert np.array_equal(out, img)


def test_recommendations_neutral():
    rec = recommend([{"hex": "#202020", "share": 0.9}], "tops")
    assert rec["suggestions"] and rec["colors"]


def test_recommendations_colourful():
    rec = recommend([{"hex": "#B4432F", "share": 0.8}], "bottoms")
    assert any("Complementary" in s or "hue" in s for s in rec["suggestions"])
