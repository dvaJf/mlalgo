import numpy as np
from gener import SineGenerator
from detect import detect
from detectai import detect_ml
import matplotlib.pyplot as plt

sg = SineGenerator(noise=0.1,n_points=500, start=0, end=8*np.pi, anomaly_count=15, anomaly_scale=2)
sg.generate()

anomalies_stat = detect(sg.df)
anomalies_ml = detect_ml(sg.df)

true_anomalies = sg.df['is_anomaly']

for name, anomalies in [('default', anomalies_stat), ('ML', anomalies_ml)]:
    detected_anomalies = anomalies
    
    tp = np.sum((true_anomalies == True) & (detected_anomalies == True))
    fp = np.sum((true_anomalies == False) & (detected_anomalies == True))
    fn = np.sum((true_anomalies == True) & (detected_anomalies == False))
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    print(f"{name}:")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1_score}")
    print()


fig, axes = plt.subplots(1, 2, figsize=(16, 5))
axes[0].plot(sg.df['x'], sg.df['y'], color='black', alpha=0.7)
axes[0].scatter(sg.df.loc[sg.df['is_anomaly'], 'x'], sg.df.loc[sg.df['is_anomaly'], 'y'], color='red', zorder=5, label='Anomalies', s=50)
axes[0].scatter(sg.df.loc[anomalies_stat, 'x'], sg.df.loc[anomalies_stat, 'y'], color='blue', zorder=5, label='default', marker='x', s=80)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(sg.df['x'], sg.df['y'], color='black', alpha=0.7)
axes[1].scatter(sg.df.loc[sg.df['is_anomaly'], 'x'], sg.df.loc[sg.df['is_anomaly'], 'y'], color='red', zorder=5, label='Anomalies', s=50)
axes[1].scatter(sg.df.loc[anomalies_ml, 'x'], sg.df.loc[anomalies_ml, 'y'], color='blue', zorder=5, label='ML', marker='x', s=80)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()