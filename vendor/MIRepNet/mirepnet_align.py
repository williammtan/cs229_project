"""Channel template + alignment utilities, vendored verbatim from
https://github.com/staraink/MIRepNet (utils/channel_list.py, utils/utils.py).

Only the pieces needed to format an arbitrary MI montage into MIRepNet's 45-channel
template are kept: the 45-ch template (`use_channels_names`), the electrode-position
table (`channel_positions`), Euclidean Alignment (`EA`), and the inverse-distance
channel interpolation (`pad_missing_channels_diff`).
"""
import numpy as np
from scipy.spatial.distance import cdist

# MIRepNet's 45-channel input template (utils/channel_list.py:use_channels_names).
use_channels_names = [
    "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8",
    "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8",
    "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8",
    "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8",
]

# Native channel names of common MOABB MI datasets (utils/channel_list.py).
BNCI2014001_chn_names = ['FZ', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'CZ',
                         'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'P1', 'PZ', 'P2', 'POZ']

# Electrode 2-D layout used for inverse-distance interpolation (utils/channel_list.py).
channel_positions = {
    'FP1': (-0.3, 0.9), 'FPZ': (0, 1.0), 'FP2': (0.3, 0.9),
    'AF7': (-0.3, 0.8), 'AF3': (-0.2, 0.8), 'AFZ': (0, 0.8), 'AF4': (0.2, 0.8), 'AF8': (0.3, 0.8),
    'F9': (-0.6, 0.7), 'F7': (-0.5, 0.7), 'F5': (-0.4, 0.7), 'F3': (-0.3, 0.7),
    'F1': (-0.15, 0.7), 'FZ': (0, 0.7), 'F2': (0.15, 0.7),
    'F4': (0.3, 0.7), 'F6': (0.4, 0.7), 'F8': (0.5, 0.7), 'F10': (0.6, 0.7),
    'FT9': (-0.7, 0.6), 'FT7': (-0.6, 0.6), 'FC5': (-0.5, 0.6), 'FC3': (-0.4, 0.6),
    'FC1': (-0.2, 0.6), 'FCZ': (0, 0.6), 'FC2': (0.2, 0.6),
    'FC4': (0.4, 0.6), 'FC6': (0.5, 0.6), 'FT8': (0.6, 0.6), 'FT10': (0.6, 0.7),
    'FTT9': (-1.1, 0.55), 'T7': (-1.0, 0.5), 'TPP7': (-0.9, 0.45), 'C5': (-0.7, 0.5), 'C3': (-0.4, 0.5),
    'C1': (-0.2, 0.5), 'CZ': (0, 0.5), 'C2': (0.2, 0.5),
    'C4': (0.4, 0.5), 'C6': (0.7, 0.5), 'TPP8': (9.0, 0.45), 'T8': (1.0, 0.5), 'FTT10': (1.1, 0.55),
    'TP9': (-0.8, 0.4), 'TPP9': (-0.7, 0.4), 'TP7': (-0.6, 0.4), 'CP5': (-0.5, 0.4), 'CP3': (-0.4, 0.4),
    'CP1': (-0.2, 0.4), 'CPZ': (0, 0.4), 'CP2': (0.2, 0.4),
    'CP4': (0.4, 0.4), 'CP6': (0.5, 0.4), 'TP8': (0.6, 0.4), 'TPP10': (0.7, 0.4), 'TP10': (0.8, 0.4),
    'P9': (-0.6, 0.3), 'P7': (-0.5, 0.3), 'P5': (-0.4, 0.3), 'P3': (-0.3, 0.3),
    'P1': (-0.15, 0.3), 'PZ': (0, 0.3), 'P2': (0.15, 0.3),
    'P4': (0.3, 0.3), 'P6': (0.4, 0.3), 'P8': (0.5, 0.3), 'P10': (0.6, 0.3),
    'PO9': (-0.5, 0.2), 'PO7': (-0.4, 0.2), 'PO5': (-0.3, 0.2), 'PO3': (-0.2, 0.2), 'POZ': (0, 0.2),
    'PO4': (0.2, 0.2), 'PO6': (0.3, 0.2), 'PO8': (0.4, 0.2), 'PO10': (0.5, 0.2),
    'O1': (-0.2, 0.1), 'OZ': (0, 0.1), 'O2': (0.2, 0.1),
}


def EA(x):
    """Euclidean Alignment (utils/utils.py:EA). x: (n_trials, n_channels, n_times)."""
    cov = np.zeros((x.shape[0], x.shape[1], x.shape[1]))
    for i in range(x.shape[0]):
        cov[i] = np.cov(x[i])
    refEA = np.mean(cov, 0)
    from scipy.linalg import fractional_matrix_power
    sqrtRefEA = fractional_matrix_power(refEA, -0.5)
    XEA = np.zeros(x.shape)
    for i in range(x.shape[0]):
        XEA[i] = np.dot(sqrtRefEA, x[i])
    return XEA


def pad_missing_channels_diff(x, target_channels, actual_channels):
    """Map actual montage -> target template via inverse-distance interpolation
    (utils/utils.py:pad_missing_channels_diff). Present channels are copied;
    missing template channels are interpolated from all present channels weighted
    by 1/distance in `channel_positions`. x: (B, C_actual, T) -> (B, C_target, T)."""
    B, C, T = x.shape
    num_target = len(target_channels)
    existing_pos = np.array([channel_positions[ch] for ch in actual_channels])
    target_pos = np.array([channel_positions[ch] for ch in target_channels])

    W = np.zeros((num_target, C))
    for i, (target_ch, pos) in enumerate(zip(target_channels, target_pos)):
        if target_ch in actual_channels:
            src_idx = actual_channels.index(target_ch)
            W[i, src_idx] = 1.0
        else:
            dist = cdist([pos], existing_pos)[0]
            weights = 1 / (dist + 1e-6)
            weights /= weights.sum()
            W[i] = weights

    padded = np.zeros((B, num_target, T))
    for b in range(B):
        padded[b] = W @ x[b]
    return padded
