 import numpy as np

def mpjpe(predicted, ground_truth):
    """
    Calculate Mean Per Joint Position Error.
    Both inputs contain 3D joint coordinates in meters.
    Returns the average error in millimeters.
    """
    predicted = np.asarray(predicted, dtype=float)
    ground_truth = np.asarray(ground_truth, dtype=float)

    if predicted.shape != ground_truth.shape:
        raise ValueError("Predicted and ground-truth shapes must match.")

    if predicted.ndim < 2 or predicted.shape[-1] != 3:
        raise ValueError("Expected joint coordinates with shape (..., J, 3).")

    distances = np.linalg.norm(predicted - ground_truth, axis=-1)
    return np.mean(distances) * 1000
