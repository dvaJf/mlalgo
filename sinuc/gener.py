import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class SineGenerator:
    def __init__(self, noise=0.1,n_points=300, anomaly_count=5, anomaly_scale=3.0, amplitude=1, frequency=1, start=0, end=4*np.pi):
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
        x = np.linspace(self.start, self.end, self.n_points)
        noise = np.random.normal(0, self.noise, size=len(x))
        y = self.amplitude * np.sin(self.frequency * x) + noise

        anomal = np.random.choice(len(x), size=self.anomaly_count)
        anomal_val = y[anomal] + np.random.choice([-1, 1], size=self.anomaly_count) * self.anomaly_scale

        self.df = pd.DataFrame({'x': x, 'y': y})
        self.df['is_anomaly'] = False
        self.df.loc[anomal, 'y'] = anomal_val
        self.df.loc[anomal, 'is_anomaly'] = True