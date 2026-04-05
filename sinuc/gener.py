import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class SineGenerator:
    def __init__(self, noise=0.1, anomaly_count=5, anomaly_scale=3.0, amplitude=1, frequency=1, start=0, end=4*np.pi):
        self.noise = noise
        self.anomaly_count = anomaly_count
        self.anomaly_scale = anomaly_scale
        self.amplitude = amplitude
        self.frequency = frequency
        self.start = start
        self.end = end
        self.df = None

    def generate(self):
        x = np.linspace(self.start, self.end, 300)
        noise = np.random.normal(0, self.noise, size=len(x))
        y = self.amplitude * np.sin(self.frequency * x) + noise

        anomal = np.random.choice(len(x), size=self.anomaly_count)
        anomal_val = y[anomal] + np.random.choice([-1, 1], size=self.anomaly_count) * self.anomaly_scale

        self.df = pd.DataFrame({'x': x, 'y': y})
        self.df['is_anomaly'] = False
        self.df.loc[anomal, 'y'] = anomal_val
        self.df.loc[anomal, 'is_anomaly'] = True

    def plot(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.df['x'], self.df['y'])
        plt.scatter(self.df.loc[self.df['is_anomaly'], 'x'], self.df.loc[self.df['is_anomaly'], 'y'], color='red', zorder=5)
        plt.xlabel('x')
        plt.ylabel('sin(x)')
        plt.show()
