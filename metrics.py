import numpy as np


def compute_metrics(true_anomalies: np.ndarray, predicted_anomalies: np.ndarray) -> dict:
    """
    Вычисляет метрики качества бинарной классификации аномалий.

    Параметры
    ----------
    true_anomalies : numpy.ndarray
        Массив истинных меток (True — аномалия, False — норма).
    predicted_anomalies : numpy.ndarray
        Массив предсказанных меток детектора.

    Возвращает
    ----------
    dict
        Словарь с ключами 'tp', 'fp', 'fn', 'precision', 'recall', 'f1'.
    """
    true_anomalies = np.asarray(true_anomalies, dtype=bool)
    predicted_anomalies = np.asarray(predicted_anomalies, dtype=bool)

    # TP: истинные аномалии, корректно найденные детектором
    tp = int(np.sum(true_anomalies & predicted_anomalies))
    # FP: обычные точки, ошибочно помеченные как аномалии
    fp = int(np.sum(~true_anomalies & predicted_anomalies))
    # FN: аномалии, которые детектор пропустил
    fn = int(np.sum(true_anomalies & ~predicted_anomalies))

    # Защита от деления на ноль
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # F1-score — гармоническое среднее Precision и Recall
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
