import numpy as np
from gener import SineGenerator
from detect import detect
from detectai import detect_ml
import matplotlib.pyplot as plt

sg = SineGenerator(noise=0.1, start=0, end=8*np.pi, anomaly_count=10, anomaly_scale=1)
sg.generate()

anomalies_stat = detect(sg.df, n_sigma=2)
anomalies_ml = detect_ml(sg.df)

true_anomalies = sg.df['is_anomaly']

for name, anomalies in [('Statistical', anomalies_stat), ('ML', anomalies_ml)]:
    detected_anomalies = anomalies
    
    true_positives = np.sum((true_anomalies == True) & (detected_anomalies == True))
    false_positives = np.sum((true_anomalies == False) & (detected_anomalies == True))
    false_negatives = np.sum((true_anomalies == True) & (detected_anomalies == False))
    
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    print(f"{name}:")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1_score}")
    print()

plt.figure(figsize=(10, 5))
plt.plot(sg.df['x'], sg.df['y'], label='Generated Data')
plt.scatter(sg.df.loc[sg.df['is_anomaly'], 'x'], sg.df.loc[sg.df['is_anomaly'], 'y'], color='red', zorder=5, label='True Anomalies')
plt.scatter(sg.df.loc[anomalies_ml, 'x'], sg.df.loc[anomalies_ml, 'y'], color='blue', zorder=5, label='Detected Anomalies')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()