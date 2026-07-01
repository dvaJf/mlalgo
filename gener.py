from __future__ import annotations
import numpy as np
import pandas as pd


SIGNAL_TYPES = {
    "Синусоида": "sine",
    "Пользовательская формула": "custom",
}


class SineGenerator:
    """Генератор рядов."""

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
        """Базовый сигнал."""
        a = self.amplitude
        f = self.frequency

        if self.signal_type == "sine":
            return a * np.sin(f * x)

        elif self.signal_type == "custom" and self.formula:
            # Формула
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
            return a * np.sin(f * x)

    def generate(self) -> pd.DataFrame:
        """Генерация."""
        rng = np.random.default_rng(self.seed)

        # X-сетка
        x = np.linspace(self.start, self.end, self.n_points)

        # Сигнал
        y_clean = self._compute_signal(x)

        # Масштаб
        y_scale = np.std(y_clean)
        if y_scale < 1e-6:
            y_scale = 1.0

        # Шум
        noise_array = rng.normal(0, self.noise * y_scale, size=len(x))

        y = y_clean + noise_array

        actual_anomaly_count = min(self.anomaly_count, len(x))

        # Аномалии
        anomal = rng.choice(len(x), size=actual_anomaly_count, replace=False)

        anomaly_shift = rng.choice([-1, 1], size=actual_anomaly_count) * (self.anomaly_scale * y_scale)
        anomal_val = y[anomal] + anomaly_shift

        # DataFrame
        self.df = pd.DataFrame({'x': x, 'y': y})
        self.df['is_anomaly'] = False

        self.df.loc[anomal, 'y'] = anomal_val
        self.df.loc[anomal, 'is_anomaly'] = True

        return self.df