"""Threshold evaluation against an input voxel grid; no model execution."""
import numpy as np


def threshold_sweep(original, probabilities, thresholds=None):
    thresholds = np.linspace(0, 1, 101) if thresholds is None else thresholds
    truth = np.asarray(original, dtype=bool)
    scores = np.asarray(probabilities)
    if truth.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError('Reference and reconstruction must be finite and have matching shapes.')
    rows = []
    for threshold in thresholds:
        predicted = scores > threshold
        tp = np.count_nonzero(truth & predicted)
        fp = np.count_nonzero(~truth & predicted)
        fn = np.count_nonzero(truth & ~predicted)
        rows.append({'Threshold': float(threshold),
                     'IoU': tp / (tp + fp + fn) if tp + fp + fn else 0.0,
                     'F1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
                     'Precision': tp / (tp + fp) if tp + fp else 0.0,
                     'Recall': tp / (tp + fn) if tp + fn else 0.0})
    return rows
