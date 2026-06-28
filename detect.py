import numpy as np
import pandas as pd


def detect(df: pd.DataFrame, window: int | None = None, n_sigma: float = 3.0, return_details: bool = False) -> np.ndarray | tuple[np.ndarray, dict]:
    """
    Статистический детектор аномалий на основе медианного Z-score (MAD).

    Медианный фильтр с маленьким окном идеально огибает сложные формы
    сигнала (включая прямоугольные волны и резкие скачки), но игнорирует
    одиночные точечные выбросы. Поэтому отклонение от локальной медианы
    — отличный универсальный детектор.

    Параметры
    ----------
    df : pandas.DataFrame
        DataFrame, содержащий колонку 'y' с значениями временного ряда.
    window : int | None
        Размер локального окна. По умолчанию 7 — оптимально для
        отслеживания быстрых изменений сигнала и игнорирования точечных выбросов.
    n_sigma : float
        Порог по Z-score (по умолчанию 3.0).

    Возвращает
    ----------
    numpy.ndarray
        Булев массив длины n, где True — аномальная точка.
    """
    n = len(df)

    # Используем маленькое окно по умолчанию
    if window is None:
        window = 7

    y = df["y"]

    # Оценка глобального шума (2-я производная)
    # 2-я разность устраняет линейные и квадратичные тренды (например, полиномы x^2).
    # Это позволяет идеально и независимо оценить уровень случайного шума в данных.
    d2y = y.diff().diff().dropna()
    # Дисперсия 2-й разности белого шума равна 6 * sigma^2. Отсюда std = 2.45 * sigma.
    # Используем робастную медиану для защиты от самих аномалий.
    global_noise_std = np.median(np.abs(d2y - np.median(d2y))) / 0.6745 / 2.45
    min_std = max(global_noise_std, 1e-6)

    # Медианный Z-score (MAD)
    # Медиана не сдвигается от аномалий внутри окна и отлично отслеживает
    # любые формы сигналов (квадраты, пилы, шумные синусоиды).
    rolling_median = y.rolling(window, center=True, min_periods=1).median()
    
    # MAD = median(|y - median(y)|) — робастная мера разброса
    mad = (y - rolling_median).abs().rolling(window, center=True, min_periods=1).median()
    
    # Коэффициент 1.4826 переводит MAD в масштаб стандартного отклонения
    mad_std = mad * 1.4826

    # Нижняя граница для локального стандартного отклонения.
    # Защищает от ложных срабатываний на участках с крутым уклоном (например, на полиномах),
    # где локальная дисперсия может ложно стремиться к нулю из-за идеального трекинга.
    mad_std_clipped = np.maximum(mad_std, min_std)

    # Z-score на основе медианы и MAD
    mad_z = ((y - rolling_median) / mad_std_clipped).abs().fillna(0)
    
    # Точка считается аномалией, если медианный Z-score превышает порог
    anomalies = (mad_z > n_sigma).to_numpy()

    if return_details:
        upper = rolling_median + mad_std_clipped * n_sigma
        lower = rolling_median - mad_std_clipped * n_sigma
        return anomalies, {'upper': upper, 'lower': lower, 'median': rolling_median}

    return anomalies
