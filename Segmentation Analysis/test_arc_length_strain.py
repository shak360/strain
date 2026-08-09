"""Weights-free unit tests for arc_length_strain.py.

Run from this directory with: python -m pytest test_arc_length_strain.py
No model weights, videos, or network access needed.
"""
from __future__ import annotations

import ast
import inspect
import os
import pathlib
import re

os.environ.setdefault("MPLBACKEND", "Agg")  # headless-safe before pyplot import

import arc_length_strain as V

HERE = pathlib.Path(__file__).resolve().parent


def sawtooth(period: int, n: int, low: float = 100.0, high: float = 130.0) -> list[float]:
    """A synthetic length curve: |/\\|-shaped beats with valleys every `period` frames."""
    half = period / 2
    return [high - (high - low) * abs(((i % period) / half) - 1) for i in range(n)]


# --- beat detection windows derive from fps ---

def test_calc_ratio_finds_beats_at_50fps():
    # 0.64 s beats sampled at 50 fps: the 32-frame spacing the constants were tuned on
    ratios = V.calc_ratio(sawtooth(32, 160), ".", "t.png", save=False, fps=50.0)
    assert len(ratios) == 4
    assert ratios == sorted(ratios)
    assert all(0.75 < r < 0.85 for r in ratios)  # valley/peak of the smoothed curve


def test_calc_ratio_finds_beats_at_low_fps():
    # The same 0.64 s beats sampled at 25 fps are 16 frames apart; the fixed
    # distance=32 window would have silently discarded them.
    ratios = V.calc_ratio(sawtooth(16, 160), ".", "t.png", save=False, fps=25.0)
    assert len(ratios) >= 3


def test_windows_scale_with_fps():
    # One physical signal, two sampling rates: same beats found either way.
    n50 = len(V.calc_ratio(sawtooth(32, 160), ".", "t.png", save=False, fps=50.0))
    n100 = len(V.calc_ratio(sawtooth(64, 320), ".", "t.png", save=False, fps=100.0))
    assert n50 == n100 == 4


def test_calc_ratio_is_deterministic():
    curve = sawtooth(32, 160)
    assert V.calc_ratio(curve, ".", "t.png", save=False) == V.calc_ratio(curve, ".", "t.png", save=False)


def test_fps_defaults():
    # estimate_strain reads fps from the container by default; the tuned rate
    # remains the explicit default further down the chain.
    assert inspect.signature(V.estimate_strain).parameters["fps"].default is None
    assert inspect.signature(V.strain_lengths).parameters["fps"].default == 50.0
    assert inspect.signature(V.calc_ratio).parameters["fps"].default == 50.0


# --- smoothing method is a caller decision, threaded end to end ---

def test_smoothing_methods_dispatch():
    curve = sawtooth(32, 160)
    for method in ("Moving Average", "ConvolveAverage", "Savgol", "FFT", "low"):
        ratios = V.calc_ratio(curve, ".", "t.png", save=False, smoothening_function=method)
        assert ratios == sorted(ratios) and len(ratios) >= 1


def test_smoothing_parameter_threads_with_one_default():
    for f in (V.estimate_strain, V.strain_lengths, V.calc_ratio):
        assert inspect.signature(f).parameters["smoothening_function"].default == "ConvolveAverage"


# --- flip defaults agree across the call chain ---

def test_flip_defaults_agree():
    outer = inspect.signature(V.estimate_strain).parameters["flip"].default
    inner = inspect.signature(V.Segmentation.single_vid_prediction).parameters["flip"].default
    assert outer is False and inner is False


# --- repo-level regression guards ---

def test_vid_to_strain_shim_still_works():
    import Vid_to_Strain
    assert Vid_to_Strain.estimate_strain is V.estimate_strain
    assert Vid_to_Strain.calc_ratio is V.calc_ratio

def test_setup_py_parses():
    ast.parse((HERE.parent / "setup.py").read_text())


def test_no_removed_numpy_int_alias():
    echo = (HERE / "echonet" / "datasets" / "echo.py").read_text()
    assert not re.search(r"np\.int\b", echo)
