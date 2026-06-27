import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler


def _compute_features(x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    """
    Вычисляет инженерные признаки для каждой точки временного ряда.

    Признаки помогают ML-модели отличать аномалии от нормального поведения сигнала.
    Используются признаки разного масштаба: локальные (производные, скользящие окна),
    глобальные (отклонения от среднего/медианы) и многомасштабные (несколько размеров окна).

    Параметры
    ----------
    x : numpy.ndarray
        Массив координат по оси X (время/аргумент).
    y : numpy.ndarray
        Массив значений временного ряда.

    Возвращает
    ----------
    dict
        Словарь с признаками, где ключ — название, значение — массив длины n.
    """
    n = len(y)
    features = {}

    # 1. Исходное значение ряда
    features["y"] = y

    # 2. Первая производная (скорость изменения)
    dy = np.gradient(y)
    features["dy"] = dy

    # 3. Абсолютное значение первой производной (резкость скачка)
    features["abs_dy"] = np.abs(dy)

    # 4. Вторая производная (ускорение / кривизна)
    features["d2y"] = np.gradient(dy)

    # 5-6. Остатки после сглаживания с разными размерами окна
    #       Несколько масштабов помогают ловить аномалии разного характера
    for w_scale in [50, 100]:
        window = max(10, n // w_scale)
        kernel = np.ones(window) / window
        y_smooth = np.convolve(y, kernel, mode="same")
        residual = y - y_smooth
        features[f"residual_{w_scale}"] = np.abs(residual)

    # 7. Локальное стандартное отклонение (скользящее окно)
    #    Аномалии создают повышенную локальную дисперсию
    local_window = max(10, n // 50)
    series = pd.Series(y)
    rolling_std = series.rolling(local_window, center=True, min_periods=1).std(ddof=0).to_numpy()
    features["rolling_std"] = rolling_std

    # 8. Локальный Z-score (насколько точка отклоняется от локального среднего)
    rolling_mean = series.rolling(local_window, center=True, min_periods=1).mean().to_numpy()
    local_zscore = np.where(rolling_std > 0, np.abs(y - rolling_mean) / rolling_std, 0)
    features["local_zscore"] = local_zscore

    # 9. Размах в локальном окне (max - min)
    rolling_max = series.rolling(local_window, center=True, min_periods=1).max().to_numpy()
    rolling_min = series.rolling(local_window, center=True, min_periods=1).min().to_numpy()
    features["rolling_range"] = rolling_max - rolling_min

    # 10. Отклонение от глобального среднего
    features["mean"] = np.abs(y - np.mean(y))

    # 11. Отклонение от глобальной медианы (устойчиво к выбросам)
    features["median"] = np.abs(y - np.median(y))

    return features


def detect_ml(df: pd.DataFrame, contamination: float | None = None) -> np.ndarray:
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

    # === 1. Формирование признаков ===
    feat_dict = _compute_features(df["x"].to_numpy(), y)
    features = np.column_stack(list(feat_dict.values()))

    # === 2. Масштабирование ===
    # RobustScaler использует медиану и IQR, игнорируя выбросы
    features_scaled = RobustScaler().fit_transform(features)

    # === 3. Обучение Isolation Forest ===
    model = IsolationForest(
        n_estimators=500,      # Число деревьев (больше = стабильнее оценка)
        max_samples=min(256, n),  # Ограничиваем подвыборку размером данных
        max_features=0.8,      # Доля признаков на каждом разбиении
        bootstrap=True,        # Бутстрап-подвыборки для разнообразия деревьев
        random_state=42,       # Фиксация seed для воспроизводимости
        n_jobs=-1              # Параллельное обучение на всех ядрах
    )
    model.fit(features_scaled)

    # === 4. Вычисление аномальности ===
    # score_samples: чем меньше значение, тем сильнее точка выделяется как аномалия
    scores = model.score_samples(features_scaled)

    # === 5. Адаптивный выбор порога ===
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

    return anomalies