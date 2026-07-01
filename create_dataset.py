import numpy as np
import pandas as pd
from gener import SineGenerator

# Число рядов
n = 100
all_data = []

for id in range(n):
    # Генератор
    sg = SineGenerator(
        n_points=3000,        # Длина
        noise=0.1,            # Шум
        anomaly_count=150,    # Аномалии
        anomaly_scale=2.0,    # Скачок
        amplitude=1.0,        # Амплитуда
        frequency=1.0,        # Частота
        start=0,
        end=64 * np.pi
    )

    sg.generate()

    # ID
    df = sg.df.copy()
    df['id'] = id

    df = df[['id', 'x', 'y', 'is_anomaly']]
    all_data.append(df)

# Объединяем
full_data = pd.concat(all_data, ignore_index=True)

# В CSV
full_data.to_csv('test_data.csv', index=False)

print(f"точек {len(full_data)}")
print(f"аномалий {full_data['is_anomaly'].sum()}")
print(f"рядов {n}")