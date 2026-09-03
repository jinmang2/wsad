"""Qualitative figure util (issue #21)."""

import numpy as np

from src.viz import _mask_to_intervals, plot_anomaly_scores


def test_mask_to_intervals():
    m = np.array([0, 1, 1, 0, 0, 1, 0])
    assert _mask_to_intervals(m) == [(1, 3), (5, 6)]


def test_mask_to_intervals_edges():
    assert _mask_to_intervals(np.array([1, 1, 0, 1])) == [(0, 2), (3, 4)]
    assert _mask_to_intervals(np.array([0, 0])) == []


def test_plot_returns_curve_of_right_length():
    scores = np.linspace(0, 1, 32)
    fig, ax = plot_anomaly_scores(scores)
    assert len(ax.get_lines()[0].get_xdata()) == 32


def test_plot_shades_interval_gt_and_saves(tmp_path):
    scores = np.random.rand(50)
    out = tmp_path / "fig.png"
    fig, ax = plot_anomaly_scores(
        scores, gt=[(10, 20)], threshold=0.5, save_path=str(out)
    )
    assert out.exists()
    assert len(ax.patches) >= 1  # axvspan polygon


def test_plot_accepts_binary_mask_gt():
    scores = np.random.rand(20)
    mask = np.zeros(20)
    mask[5:10] = 1
    fig, ax = plot_anomaly_scores(scores, gt=mask)
    assert len(ax.patches) >= 1
