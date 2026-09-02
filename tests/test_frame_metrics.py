"""The secondary evaluation metrics (WSAD_INTEGRATION_PLAN.md §8).

Frame ROC-AUC is the number every paper reports, but the UCF-Crime test set is dominated
by normal frames, so a model that only separates normal *videos* from anomalous ones
already scores well. These two metrics remove that escape hatch, and the tests below are
built around exactly that failure mode.
"""

import numpy as np

from src.eval_matrix import frame_metrics


def test_primary_metrics_only_when_video_split_is_unknown():
    preds = np.array([0.1, 0.9, 0.2, 0.8])
    labels = np.array([0.0, 1.0, 0.0, 1.0])
    m = frame_metrics(preds, labels)
    assert set(m) == {"roc_auc", "pr_auc"}
    assert m["roc_auc"] == 1.0


def test_abnormal_auc_exposes_a_video_level_only_detector():
    """A model that scores every anomalous video high and every normal video low gets a
    perfect frame AUC while localizing nothing. `abnormal_auc` must catch that."""
    # 2 anomalous videos (4 frames each, only the last 2 truly anomalous) + 6 normal ones,
    # so normal frames dominate the way they do in the real test set
    labels = np.array([0, 0, 1, 1, 0, 0, 1, 1] + [0] * 24, dtype=float)
    is_abnormal = np.array([True] * 8 + [False] * 24)
    video_level = np.array([0.9] * 8 + [0.1] * 24)  # constant within each video

    m = frame_metrics(video_level, labels, is_abnormal)
    assert m["roc_auc"] > 0.9, "video-level separation alone should score well overall"
    assert m["abnormal_auc"] == 0.5, "but it localizes nothing inside anomalous videos"

    localizing = np.array([0.1, 0.1, 0.9, 0.9, 0.1, 0.1, 0.9, 0.9] + [0.1] * 24)
    assert frame_metrics(localizing, labels, is_abnormal)["abnormal_auc"] == 1.0


def test_far_normal_counts_only_normal_video_frames():
    labels = np.array([0, 1, 0, 0], dtype=float)
    is_abnormal = np.array([True, True, False, False])
    # the abnormal video's high score must not count as a false alarm
    m = frame_metrics(np.array([0.9, 0.9, 0.9, 0.1]), labels, is_abnormal)
    assert m["far_normal"] == 0.5  # 1 of the 2 normal-video frames is above 0.5


def test_far_normal_respects_the_threshold():
    labels = np.zeros(4)
    is_abnormal = np.zeros(4, dtype=bool)
    preds = np.array([0.2, 0.4, 0.6, 0.8])
    assert frame_metrics(preds, labels, is_abnormal, far_threshold=0.5)["far_normal"] == 0.5
    assert frame_metrics(preds, labels, is_abnormal, far_threshold=0.3)["far_normal"] == 0.75


def test_abnormal_auc_is_omitted_when_it_would_be_undefined():
    """Anomalous videos that happen to carry only one label class cannot yield an AUC;
    the key must be absent rather than NaN, so downstream rounding does not crash."""
    labels = np.array([1.0, 1.0, 0.0, 0.0])
    is_abnormal = np.array([True, True, False, False])
    m = frame_metrics(np.array([0.9, 0.8, 0.2, 0.1]), labels, is_abnormal)
    assert "abnormal_auc" not in m
    assert "far_normal" in m
