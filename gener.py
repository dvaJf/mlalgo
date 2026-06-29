from __future__ import annotations
import numpy as np
import pandas as pd


# Доступные типы сигналов для генерации
SIGNAL_TYPES = {
    "Синусоида": "sine",
    "Пользовательская формула": "custom",
}


class SineGenerator:
    """
    Генератор синтетических временных рядов с шумом и точечными аномалиями.

    Поддерживает несколько типов базового сигнала: синусоида, косинусоида,
    прямоугольная волна, пилообразная, треугольная, комбинированная,
    а также пользовательскую формулу.

    Параметры
    ----------
    noise : float
        Стандартное отклонение нормального шума, накладываемого на сигнал.
    n_points : int
        Количество точек в генерируемом ряде.
    anomaly_count : int
        Число аномальных точек, которые будут случайно вставлены в ряд.
    anomaly_scale : float
        Амплитуда скачка аномалии (множитель к случайному знаку +/-1).
    amplitude : float
        Амплитуда исходного сигнала.
    frequency : float
        Частота сигнала.
    start : float
        Начальное значение аргумента x.
    end : float
        Конечное значение аргумента x.
    seed : int | None
        Seed для генератора случайных чисел. Если задан, результаты воспроизводимы.
    signal_type : str
        Тип базового сигнала: 'sine', 'cosine', 'square', 'sawtooth',
        'triangle', 'combined', 'custom'.
    formula : str | None
        Математическая формула для signal_type='custom'.
        Доступные переменные: x, amplitude (a), frequency (f).
        Доступные функции: sin, cos, tan, exp, log, sqrt, abs, pi.
        Пример: 'a * sin(f * x) + 0.5 * cos(2 * f * x)'
    """

    def __init__(self, noise: float = 0.1, n_points: int = 300,
                 anomaly_count: int = 5, anomaly_scale: float = 3.0,
                 amplitude: float = 1.0, frequency: float = 1.0,
                 start: float = 0.0, end: float = 4 * np.pi,
                 seed: int | None = None,
                 signal_type: str = "sine",
                 formula: str | None = None):
        # Инициализация параметров генерации
        self.noise = noise
        self.anomaly_count = anomaly_count
        self.anomaly_scale = anomaly_scale
        self.amplitude = amplitude
        self.frequency = frequency
        self.start = start
        self.end = end
        self.n_points = n_points
        self.seed = seed
        self.signal_type = signal_type
        self.formula = formula
        self.df: pd.DataFrame | None = None

    def _compute_signal(self, x: np.ndarray) -> np.ndarray:
        """
        Вычисляет базовый сигнал (без шума и аномалий) по заданному типу.

        Параметры
        ----------
        x : numpy.ndarray
            Массив координат по оси X.

        Возвращает
        ----------
        numpy.ndarray
            Массив значений базового сигнала.
        """
        a = self.amplitude
        f = self.frequency

        if self.signal_type == "sine":
            return a * np.sin(f * x)

        elif self.signal_type == "custom" and self.formula:
            # Пользовательская формула
            # Безопасное окружение: только математические функции
            safe_namespace = {
                "x": x, "a": a, "f": f,
                "amplitude": a, "frequency": f,
                "sin": np.sin, "cos": np.cos, "tan": np.tan,
                "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
                "abs": np.abs, "pi": np.pi, "e": np.e,
                "np": np,
            }
            try:
                result = eval(self.formula, {"__builtins__": {}}, safe_namespace)
                return np.asarray(result, dtype=float)
            except Exception as err:
                raise ValueError(
                    f"Ошибка в формуле '{self.formula}': {err}"
                ) from err
        else:
            # Fallback — синусоида
            return a * np.sin(f * x)

    def generate(self) -> pd.DataFrame:
        """
        Генерирует временной ряд и сохраняет результат в self.df.

        Алгоритм:
        1. Создает равномерную сетку x от start до end.
        2. Вычисляет базовый сигнал по выбранному типу.
        3. Добавляет гауссов шум.
        4. Случайным образом выбирает anomaly_count точек и смещает их
           на +/-anomaly_scale (симулируя точечные аномалии).
        5. Формирует DataFrame с флагом is_anomaly.

        Возвращает
        ----------
        pandas.DataFrame
            DataFrame с колонками 'x', 'y', 'is_anomaly'.
        """
        # Фиксация seed для воспроизводимости (если задан)
        rng = np.random.default_rng(self.seed)

        # Равномерная сетка по оси X
        x = np.linspace(self.start, self.end, self.n_points)

        # Вычисляем чистый сигнал без шума
        y_clean = self._compute_signal(x)

        # Определяем масштаб сигнала (стандартное отклонение)
        # Это нужно, чтобы ползунки "Шум" и "Амплитуда аномалии" работали 
        # пропорционально размеру сигнала, а не в абсолютных величинах.
        # Иначе для сигнала x**3 (до 10,000,000) аномалия размером 2.0 будет невидимой.
        y_scale = np.std(y_clean)
        if y_scale < 1e-6:
            y_scale = 1.0

        # Гауссов шум, пропорциональный масштабу сигнала
        noise_array = rng.normal(0, self.noise * y_scale, size=len(x))

        # Базовый сигнал: чистый сигнал + шум
        y = y_clean + noise_array

        # Ограничиваем число аномалий количеством точек
        actual_anomaly_count = min(self.anomaly_count, len(x))

        # Случайный выбор индексов для аномалий (без повторений)
        anomal = rng.choice(len(x), size=actual_anomaly_count, replace=False)

        # Смещение аномальных точек: случайный знак +/-1 * anomaly_scale * y_scale
        anomaly_shift = rng.choice([-1, 1], size=actual_anomaly_count) * (self.anomaly_scale * y_scale)
        anomal_val = y[anomal] + anomaly_shift

        # Создание DataFrame с чистыми данными
        self.df = pd.DataFrame({'x': x, 'y': y})
        self.df['is_anomaly'] = False

        # Запись аномальных значений и установка флага
        self.df.loc[anomal, 'y'] = anomal_val
        self.df.loc[anomal, 'is_anomaly'] = True

        return self.df