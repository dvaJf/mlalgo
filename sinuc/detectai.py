import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler


def _compute_features(x, y):
    """
    Вычисляет инженерные признаки для каждой точки временного ряда.

    Признаки помогают ML-модели отличать аномалии от нормального поведения сигнала.

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

    # 3. Вторая производная (ускорение / кривизна)
    features["d2y"] = np.gradient(dy)

    # 4. Остаток после сглаживания скользящим средним
    #    Отклонение от локального тренда помогает найти выбросы
    window = max(10, n // 100)
    y_smooth = np.convolve(y, np.ones(window) / window, mode="same")
    residual = y - y_smooth
    features["window"] = np.abs(residual)

    # 5. Отклонение от глобального среднего
    features["mean"] = np.abs(y - np.mean(y))

    # 6. Отклонение от глобальной медианы (устойчиво к выбросам)
    features["median"] = np.abs(y - np.median(y))

    return features


def detect_ml(df):
    """
    ML-детектор аномалий на основе Isolation Forest.

    Алгоритм:
    1. Извлекает инженерные признаки из ряда (значения, производные,
       остатки после сглаживания, отклонения от среднего/медианы).
    2. Масштабирует признаки с помощью RobustScaler (устойчив к выбросам).
    3. Обучает IsolationForest на всем ряде (unsupervised).
    4. Выбирает порог отсечения автоматически методом "колена" (knee method):
       ищет точку наибольшего изменения плотности аномальности.
    5. Помечает точки с наименьшими score (наиболее изолированные) как аномалии.
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
        n_estimators=400,      # Число деревьев (чем больше, тем стабильнее оценка)
        max_samples=256,       # Размер подвыборки для каждого дерева
        max_features=0.8,      # Доля признаков на каждом разбиении
        bootstrap=True,        # Бутстрап-подвыборки для разнообразия деревьев
        random_state=42,       # Фиксация seed для воспроизводимости
        n_jobs=-1              # Параллельное обучение на всех ядрах
    )
    model.fit(features_scaled)

    # === 4. Вычисление аномальности ===
    # score_samples: чем меньше значение, тем сильнее точка выделяется как аномалия
    scores = model.score_samples(features_scaled)

    # === 5. Автоматический выбор порога (knee / elbow method) ===
    # Сортируем score по возрастанию: левые точки — наиболее аномальные
    sortscore = np.sort(scores)

    # Анализируем участок от 1% до 30% наиболее аномальных точек
    start_idx = int((n - 1) * 0.01)
    end_idx = int((n - 1) * 0.30) + 1
    diffs = np.diff(sortscore[start_idx:end_idx])

    if len(diffs) > 1:
        # Сглаживание разностей простым скользящим средним (окно=3)
        smoothed_diffs = np.convolve(diffs, np.ones(3) / 3, mode='valid')
        # Точка "колена" — где прирост плотности score резко замедляется
        knee_idx = np.argmax(smoothed_diffs) + 1
    else:
        # Fallback: фиксированный процент
        knee_idx = int((n - 1) * 0.30)

    # Порог: все точки с score <= threshold считаются аномалиями
    threshold = sortscore[knee_idx]
    anomalies = scores <= threshold

    return anomalies