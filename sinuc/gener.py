import numpy as np
import pandas as pd


class SineGenerator:
    """
    Генератор синтетических временных рядов на основе синусоиды с шумом и точечными аномалиями.

    noise : float
        Стандартное отклонение нормального шума, накладываемого на сигнал.
    n_points : int
        Количество точек в генерируемом ряде.
    anomaly_count : int
        Число аномальных точек, которые будут случайно вставлены в ряд.
    anomaly_scale : float
        Амплитуда скачка аномалии (множитель к случайному знаку ±1).
    amplitude : float
        Амплитуда исходной синусоиды.
    frequency : float
        Частота синусоиды.
    start : float
        Начальное значение аргумента x.
    end : float
        Конечное значение аргумента x.
    """

    def __init__(self, noise=0.1, n_points=300, anomaly_count=5, anomaly_scale=3.0,
                 amplitude=1, frequency=1, start=0, end=4 * np.pi):
        # Инициализация параметров генерации
        self.noise = noise
        self.anomaly_count = anomaly_count
        self.anomaly_scale = anomaly_scale
        self.amplitude = amplitude
        self.frequency = frequency
        self.start = start
        self.end = end
        self.n_points = n_points
        self.df = None

    def generate(self):
        """
        Генерирует временной ряд и сохраняет результат в self.df.

        Алгоритм:
        1. Создает равномерную сетку x от start до end.
        2. Строит чистую синусоиду с заданной амплитудой и частотой.
        3. Добавляет гауссов шум.
        4. Случайным образом выбирает anomaly_count точек и смещает их
           на ±anomaly_scale (симулируя точечные аномалии).
        5. Формирует DataFrame с флагом is_anomaly.
        """
        # Равномерная сетка по оси X
        x = np.linspace(self.start, self.end, self.n_points)

        # Гауссов шум с нулевым средним и заданным стандартным отклонением
        noise = np.random.normal(0, self.noise, size=len(x))

        # Базовый сигнал: синусоида + шум
        y = self.amplitude * np.sin(self.frequency * x) + noise

        # Случайный выбор индексов для аномалий (без повторений)
        anomal = np.random.choice(len(x), size=self.anomaly_count, replace=False)

        # Смещение аномальных точек: случайный знак ±1 * anomaly_scale
        anomal_val = y[anomal] + np.random.choice([-1, 1], size=self.anomaly_count) * self.anomaly_scale

        # Создание DataFrame с чистыми данными
        self.df = pd.DataFrame({'x': x, 'y': y})
        self.df['is_anomaly'] = False

        # Запись аномальных значений и установка флага
        self.df.loc[anomal, 'y'] = anomal_val
        self.df.loc[anomal, 'is_anomaly'] = True