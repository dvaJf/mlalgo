import numpy as np
import pandas as pd


def detect(df: pd.DataFrame, window: int | None = None, n_sigma: float = 2.5) -> np.ndarray:
    """
    Статистический детектор аномалий на основе скользящего окна и Z-score.

    Для каждой точки вычисляется локальное среднее и стандартное отклонение
    в окрестности (окно вокруг точки). Если нормализованное отклонение точки
    от локального среднего превышает порог n_sigma, точка помечается как аномалия.

    Дополнительно используется медианный Z-score (MAD — Median Absolute Deviation),
    который устойчив к выбросам внутри окна и ловит аномалии, которые
    исказили локальное среднее и стандартное отклонение.

    Параметры
    ----------
    df : pandas.DataFrame
        DataFrame, содержащий колонку 'y' с значениями временного ряда.
    window : int | None
        Размер локального окна (в точках). Если None, выбирается автоматически
        как max(20, n // 50), где n — длина ряда.
    n_sigma : float
        Порог по Z-score. Чем больше значение, тем меньше точек будет признано
        аномальными (консервативнее детектор).

    Возвращает
    ----------
    numpy.ndarray
        Булев массив длины n, где True — аномальная точка.
    """
    n = len(df)

    # Автоматический подбор размера окна, если не задан явно
    if window is None:
        window = max(20, n // 50)

    y = df["y"]

    # === Метод 1: Классический Z-score на скользящем окне ===
    rolling_mean = y.rolling(window, center=True, min_periods=1).mean()
    rolling_std = y.rolling(window, center=True, min_periods=1).std(ddof=0)

    z_scores = ((y - rolling_mean) / rolling_std.replace(0, np.nan)).abs().fillna(0)
    anomalies_zscore = z_scores > n_sigma

    # === Метод 2: Медианный Z-score (MAD) — устойчив к выбросам ===
    # Медиана не сдвигается от аномалий внутри окна, поэтому
    # этот метод ловит аномалии, которые «испортили» среднее
    rolling_median = y.rolling(window, center=True, min_periods=1).median()
    # MAD = median(|y - median(y)|) — робастная мера разброса
    mad = (y - rolling_median).abs().rolling(window, center=True, min_periods=1).median()
    # Коэффициент 1.4826 переводит MAD в масштаб стандартного отклонения
    mad_std = mad * 1.4826

    mad_z = ((y - rolling_median) / mad_std.replace(0, np.nan)).abs().fillna(0)
    # Для MAD используем повышенный порог (MAD чувствительнее стандартного Z-score)
    anomalies_mad = mad_z > (n_sigma + 0.5)

    # Объединение: точка считается аномалией, если хотя бы один метод её нашёл
    # При этом MAD с повышенным порогом отсекает ложные срабатывания
    anomalies = (anomalies_zscore | anomalies_mad).to_numpy()

    return anomalies