from __future__ import annotations
import numpy as np
import pandas as pd


def detect(df: pd.DataFrame, window: int | None = None, n_sigma: float = 3.0, return_details: bool = False) -> np.ndarray | tuple[np.ndarray, dict]:
    """Детектор по медианному Z-score."""
    n = len(df)

    if window is None:
        window = 7

    y = df["y"]

    # Оценка шума
    d2y = y.diff().diff().dropna()
    global_noise_std = np.median(np.abs(d2y - np.median(d2y))) / 0.6745 / 2.45
    min_std = max(global_noise_std, 1e-6)

    # MAD
    rolling_median = y.rolling(window, center=True, min_periods=1).median()
    mad = (y - rolling_median).abs().rolling(window, center=True, min_periods=1).median()
    mad_std = mad * 1.4826

    # Ограничение std
    mad_std_clipped = np.maximum(mad_std, min_std)

    # Z-score
    mad_z = ((y - rolling_median) / mad_std_clipped).abs().fillna(0)
    anomalies = (mad_z > n_sigma).to_numpy()

    if return_details:
        upper = rolling_median + mad_std_clipped * n_sigma
        lower = rolling_median - mad_std_clipped * n_sigma
        return anomalies, {'upper': upper, 'lower': lower, 'median': rolling_median}

    return anomalies
