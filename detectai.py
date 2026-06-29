from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

def _compute_features(x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    """
    Вычисляет инженерные признаки, которые стабильно работают на ЛЮБЫХ формах сигнала.
    Вместо производных (которые ломаются на скачках) используем остатки от
    медианного фильтра с маленьким окном. Медианный фильтр идеально огибает 
    любые резкие фронты (как у прямоугольной волны), но игнорирует точечные выбросы.
    """
    n = len(y)
    features = {}
    series = pd.Series(y)

    # 1. Исходное значение
    features["y"] = y

    # 2. Разница с соседями (насколько точка выбивается из окружения)
    features["diff_prev"] = (series - series.shift(1).bfill()).to_numpy()
    features["diff_next"] = (series - series.shift(-1).ffill()).to_numpy()
    
    # 3. Медианные остатки с маленькими окнами (3, 5, 11)
    # Это главный секрет: медиана игнорирует точечный выброс, но сохраняет форму сигнала!
    for w in [3, 5, 11]:
        rolling_med = series.rolling(w, center=True, min_periods=1).median().to_numpy()
        features[f"median_res_{w}"] = np.abs(y - rolling_med)

    # 4. Локальный Z-score на основе медианы (MAD) в малом окне
    local_window = 11
    rolling_med11 = series.rolling(local_window, center=True, min_periods=1).median()
    mad = (series - rolling_med11).abs().rolling(local_window, center=True, min_periods=1).median()
    mad_std = mad * 1.4826
    local_mad_zscore = np.where(mad_std > 0, np.abs(series - rolling_med11) / mad_std, 0)
    features["local_mad_zscore"] = local_mad_zscore

    # 5. Глобальные отклонения
    features["mean"] = np.abs(y - np.mean(y))
    features["median"] = np.abs(y - np.median(y))

    return features


def detect_ml(df: pd.DataFrame, contamination: float | None = None, return_details: bool = False) -> np.ndarray | tuple[np.ndarray, dict]:
    """
    ML-детектор аномалий на основе Isolation Forest.

    Алгоритм:
    1. Извлекает расширенный набор инженерных признаков из ряда.
    2. Масштабирует признаки с помощью RobustScaler (устойчив к выбросам).
    3. Обучает IsolationForest на всем ряде (unsupervised).
    4. Определяет порог отсечения на основе процентильного метода,
       который адаптируется к ожидаемой доле аномалий.
    5. Помечает точки с наименьшими score (наиболее изолированные) как аномалии.

    Параметры
    ----------
    df : pandas.DataFrame
        DataFrame с колонками 'x' и 'y'.
    contamination : float | None
        Ожидаемая доля аномалий (0..1). Если None, используется порог
        на основе адаптивного процентильного метода.

    Возвращает
    ----------
    numpy.ndarray
        Булев массив длины n, где True — аномальная точка.
    """
    y = df["y"].to_numpy()
    n = len(y)

    # 1. Формирование признаков
    feat_dict = _compute_features(df["x"].to_numpy(), y)
    features = np.column_stack(list(feat_dict.values()))

    # 2. Масштабирование
    # RobustScaler использует медиану и IQR, игнорируя выбросы
    features_scaled = RobustScaler().fit_transform(features)

    # 3. Обучение Isolation Forest
    model = IsolationForest(
        n_estimators=500,      # Число деревьев (больше = стабильнее оценка)
        max_samples=min(256, n),  # Ограничиваем подвыборку размером данных
        max_features=0.8,      # Доля признаков на каждом разбиении
        bootstrap=True,        # Бутстрап-подвыборки для разнообразия деревьев
        random_state=42,       # Фиксация seed для воспроизводимости
        n_jobs=-1              # Параллельное обучение на всех ядрах
    )
    model.fit(features_scaled)

    # 4. Вычисление аномальности
    # score_samples: чем меньше значение, тем сильнее точка выделяется как аномалия
    scores = model.score_samples(features_scaled)

    # 5. Адаптивный выбор порога
    if contamination is not None:
        # Если доля аномалий задана явно — используем процентиль
        threshold = np.percentile(scores, contamination * 100)
    else:
        # Адаптивный метод: ищем разрыв в распределении scores
        # Сортируем по возрастанию (левые = самые аномальные)
        sorted_scores = np.sort(scores)

        # Анализируем разности между соседними score
        # Ищем в диапазоне от 1% до 25% — там обычно проходит граница
        lo = max(1, int(n * 0.01))
        hi = int(n * 0.25)
        diffs = np.diff(sorted_scores[lo:hi])

        if len(diffs) > 2:
            # Сглаживаем разности скользящим средним (окно=5)
            kernel_size = min(5, len(diffs))
            smoothed = np.convolve(diffs, np.ones(kernel_size) / kernel_size, mode='valid')
            # Точка максимального скачка — граница между аномалиями и нормой
            knee_pos = lo + np.argmax(smoothed) + kernel_size // 2
            threshold = sorted_scores[knee_pos]
        else:
            # Fallback: верхние 5% считаются аномалиями
            threshold = np.percentile(scores, 5)

    anomalies = scores <= threshold

    if return_details:
        return anomalies, {'scores': scores, 'threshold': threshold}

    return anomalies