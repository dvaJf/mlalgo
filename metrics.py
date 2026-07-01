import numpy as np


def compute_metrics(true_anomalies: np.ndarray, predicted_anomalies: np.ndarray) -> dict:
    """Метрики."""
    true_anomalies = np.asarray(true_anomalies, dtype=bool)
    predicted_anomalies = np.asarray(predicted_anomalies, dtype=bool)

    tp = int(np.sum(true_anomalies & predicted_anomalies))
    fp = int(np.sum(~true_anomalies & predicted_anomalies))
    fn = int(np.sum(true_anomalies & ~predicted_anomalies))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }
