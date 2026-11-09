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

def pa_mpjpe(predicted, ground_truth):
    """
    Calculate Procrustes-Aligned MPJPE.
    Align predicted joints using rotation, translation, and scale.
    Coordinates are in meters; result is in millimeters.
    """
    predicted = np.asarray(predicted, dtype=float)
    ground_truth = np.asarray(ground_truth, dtype=float)

    if predicted.shape != ground_truth.shape:
        raise ValueError("Shapes must match.")

    if predicted.ndim != 2 or predicted.shape[1] != 3:
        raise ValueError("Expected shape (J, 3).")

    if predicted.shape[0] < 3:
        raise ValueError("At least 3 joints are required.")

    # Center the poses
    X = predicted - predicted.mean(axis=0)
    Y = ground_truth - ground_truth.mean(axis=0)

    # Find the best rotation
    U, _, Vt = np.linalg.svd(X.T @ Y)
    D = np.eye(3)
    D[-1, -1] = np.linalg.det(U @ Vt)
    R = U @ D @ Vt

    # Find the best scale
    denominator = np.sum(X ** 2)
    if denominator < 1e-12:
        raise ValueError("Predicted joints have zero variance.")

    scale = np.sum((X @ R) * Y) / denominator

    # Align and calculate error
    aligned = scale * (X @ R) + ground_truth.mean(axis=0)

    return np.mean(
        np.linalg.norm(aligned - ground_truth, axis=1)
    ) * 1000
