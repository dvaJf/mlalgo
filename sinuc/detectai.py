import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler


def _compute_features(x, y):
    """Вычисляет расширенный набор признаков для обнаружения аномалий."""
    n = len(y)
    features = {}

    # === Базовые признаки ===
    features["x"] = x
    features["y"] = y

    # === Производные признаки ===
    dy = np.gradient(y)
    d2y = np.gradient(dy)
    features["dy"] = dy
    features["d2y"] = d2y

    # === Скользящее среднее и остаток (адаптивное окно) ===
    window = max(5, n // 20)
    kernel = np.ones(window) / window
    y_smooth = np.convolve(y, kernel, mode="same")
    residual = y - y_smooth
    features["residual"] = np.abs(residual)

    # === Локальная статистика ===
    half_w = window // 2
    local_std = np.array([
        np.std(y[max(0, i - half_w):min(n, i + half_w + 1)])
        for i in range(n)
    ])
    local_max = np.array([
        np.max(y[max(0, i - half_w):min(n, i + half_w + 1)])
        for i in range(n)
    ])
    local_min = np.array([
        np.min(y[max(0, i - half_w):min(n, i + half_w + 1)])
        for i in range(n)
    ])
    local_range = local_max - local_min

    features["local_std"] = local_std
    features["local_range"] = local_range

    # === Отклонение от локальной статистики ===
    features["zscore_local"] = np.abs(y - y_smooth) / (local_std + 1e-8)
    features["deviation_from_range"] = np.abs(y - y_smooth) / (local_range + 1e-8)

    # === Соотношение производных ===
    y_mean = np.abs(y).mean() + 1e-8
    features["dy_ratio"] = np.abs(dy) / y_mean
    features["d2y_ratio"] = np.abs(d2y) / y_mean

    # === Кумулятивное отклонение (накопленное) ===
    cum_residual = np.cumsum(np.abs(residual))
    features["cum_residual"] = cum_residual / (cum_residual[-1] + 1e-8)

    # === Сглаженный residual на разных масштабах ===
    for w_mult in [3, 7, 15]:
        w = max(3, min(w_mult, n // 3))
        k = np.ones(w) / w
        ys = np.convolve(y, k, mode="same")
        features[f"residual_w{w}"] = np.abs(y - ys)

    # === Экспоненциальное сглаживание ===
    alpha = 2.0 / (window + 1)
    y_ema = np.zeros(n)
    y_ema[0] = y[0]
    for i in range(1, n):
        y_ema[i] = alpha * y[i] + (1 - alpha) * y_ema[i - 1]
    features["ema_residual"] = np.abs(y - y_ema)

    # === Отклонение от медианы ===
    median_y = np.median(y)
    features["median_dev"] = np.abs(y - median_y)

    # === Локальная медиана (скользящая) ===
    local_median = np.array([
        np.median(y[max(0, i - half_w):min(n, i + half_w + 1)])
        for i in range(n)
    ])
    features["local_median_dev"] = np.abs(y - local_median)

    # === Амплитуда относительно локального диапазона ===
    features["amplitude_ratio"] = np.abs(y - y_smooth) / (local_range + 1e-8)

    # === Скорость изменения остатка ===
    d_residual = np.gradient(np.abs(residual))
    features["d_residual"] = np.abs(d_residual)

    return features


def detect_ml(df, contamination=0.05):
    """
    Обнаружение аномалий с помощью гибридного подхода:
    ансамбль Isolation Forest + статистический фильтр.
    
    Параметры:
        df: DataFrame с колонками 'x' и 'y'
        contamination: ожидаемая доля аномалий
    
    Возвращает:
        numpy array булевых значений (True = аномалия)
    """
    x = df["x"].to_numpy(dtype=np.float64)
    y = df["y"].to_numpy(dtype=np.float64)

    feat_dict = _compute_features(x, y)
    feature_names = list(feat_dict.keys())
    features = np.column_stack([feat_dict[name] for name in feature_names])

    scaler = RobustScaler()
    features_scaled = scaler.fit_transform(features)

    # === Ансамбль из нескольких Isolation Forest ===
    configs = [
        {"n_estimators": 500, "max_samples": 256, "max_features": 0.8, "random_state": 42},
        {"n_estimators": 300, "max_samples": 128, "max_features": 0.6, "random_state": 123},
        {"n_estimators": 400, "max_samples": 200, "max_features": 0.7, "random_state": 456},
        {"n_estimators": 600, "max_samples": 256, "max_features": 0.5, "random_state": 789},
        {"n_estimators": 350, "max_samples": 180, "max_features": 0.9, "random_state": 101},
    ]

    n = len(y)
    ensemble_anomaly_count = np.zeros(n)

    for cfg in configs:
        model = IsolationForest(
            contamination=contamination,
            bootstrap=True,
            n_jobs=-1,
            **cfg
        )
        preds = model.fit_predict(features_scaled)
        # preds == -1 означает аномалию
        ensemble_anomaly_count += (preds == -1).astype(int)

    # Голосование: если хотя бы 2 модели считают аномалией → аномалия
    ml_anomalies = ensemble_anomaly_count >= 2

    # === Статистический фильтр: z-score от глобального residual ===
    residual = feat_dict["residual"]
    # Также проверяем: точки с высоким остатком и резким градиентом
    dy = feat_dict["dy"]
    d2y = feat_dict["d2y"]

    # Комбинированный скор: остаток + производные
    combined_score = (
        0.5 * residual / (residual.std() + 1e-8) +
        0.3 * np.abs(dy) / (np.abs(dy).std() + 1e-8) +
        0.2 * np.abs(d2y) / (np.abs(d2y).std() + 1e-8)
    )

    # Статистический порог: top contamination% по комбинированному скору
    stat_threshold = np.percentile(combined_score, (1 - contamination) * 100)
    stat_anomalies = combined_score > stat_threshold

    # === Объединяем: OR (ансамбль ИЛИ статистика) для увеличения recall ===
    # Но добавляем дополнительную проверку: аномалия должна быть значимой
    # по хотя бы одному признаку
    final_anomalies = ml_anomalies | stat_anomalies

    # Дополнительная фильтрация: убираем точки, которые выглядят как обычный шум
    # — если residual < медиана * 1.5 , то не аномалия
    median_residual = np.median(residual)
    too_low_residual = residual < median_residual * 1.5
    # Отменяем для тех, кто прошёл ML ансамбль (надёжные)
    final_anomalies[too_low_residual & ~ml_anomalies] = False

    return final_anomalies