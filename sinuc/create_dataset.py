import numpy as np
import pandas as pd
from gener import SineGenerator

n = 100
all_data = []

for id in range(n):
    sg = SineGenerator(
        n_points=3000,
        noise= 0.1,
        anomaly_count=150,
        anomaly_scale=2.0,
        amplitude=1.0,
        frequency=1.0,
        start=0,
        end=64 * np.pi
    )
    sg.generate()
    df = sg.df.copy()
    df['id'] = id
    df = df[['id', 'x', 'y', 'is_anomaly']]
    all_data.append(df)

full_data = pd.concat(all_data, ignore_index=True)
full_data.to_csv('test_data.csv', index=False)

print(f"точек {len(full_data)}")
print(f"аномалий {full_data['is_anomaly'].sum()}")
print(f"рядов {n}")