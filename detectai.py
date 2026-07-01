from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

def _compute_features(x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    """Признаки для детектирования."""
    n = len(y)
    features = {}
    series = pd.Series(y)

    features["y"] = y

    # Разница
    features["diff_prev"] = (series - series.shift(1).bfill()).to_numpy()
    features["diff_next"] = (series - series.shift(-1).ffill()).to_numpy()
    
    # Окна (3, 5, 11)
    for w in [3, 5, 11]:
        rolling_med = series.rolling(w, center=True, min_periods=1).median().to_numpy()
        features[f"median_res_{w}"] = np.abs(y - rolling_med)

    # Локальный Z-score
    local_window = 11
    rolling_med11 = series.rolling(local_window, center=True, min_periods=1).median()
    mad = (series - rolling_med11).abs().rolling(local_window, center=True, min_periods=1).median()
    mad_std = mad * 1.4826
    local_mad_zscore = np.where(mad_std > 0, np.abs(series - rolling_med11) / mad_std, 0)
    features["local_mad_zscore"] = local_mad_zscore

    # Отклонения
    features["mean"] = np.abs(y - np.mean(y))
    features["median"] = np.abs(y - np.median(y))

    return features


def detect_ml(df: pd.DataFrame, contamination: float | None = None, return_details: bool = False) -> np.ndarray | tuple[np.ndarray, dict]:
    """ML-детектор (Isolation Forest)."""
    y = df["y"].to_numpy()
    n = len(y)

    # Признаки
    feat_dict = _compute_features(df["x"].to_numpy(), y)
    features = np.column_stack(list(feat_dict.values()))

    # Scaler
    features_scaled = RobustScaler().fit_transform(features)

    # Модель
    model = IsolationForest(
        n_estimators=500,
        max_samples=min(256, n),
        max_features=0.8,
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )
    model.fit(features_scaled)

    # Оценка
    scores = model.score_samples(features_scaled)

    # Порог
    if contamination is not None:
        threshold = np.percentile(scores, contamination * 100)
    else:
        sorted_scores = np.sort(scores)

        lo = max(1, int(n * 0.01))
        hi = int(n * 0.25)
        diffs = np.diff(sorted_scores[lo:hi])

        if len(diffs) > 2:
            kernel_size = min(5, len(diffs))
            smoothed = np.convolve(diffs, np.ones(kernel_size) / kernel_size, mode='valid')
            knee_pos = lo + np.argmax(smoothed) + kernel_size // 2
            threshold = sorted_scores[knee_pos]
        else:
            threshold = np.percentile(scores, 5)

    anomalies = scores <= threshold

    if return_details:
        return anomalies, {'scores': scores, 'threshold': threshold}

    return anomalies