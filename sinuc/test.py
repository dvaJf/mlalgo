import numpy as np
import pandas as pd
from detect import detect

df = pd.read_csv('test_data.csv')
ids = df['id'].unique()

results = []
all_predictions = []

for id in ids:
    df_series = df[df['id'] == id].copy()
    predictions = detect(df_series)
    true_anomalies = df_series['is_anomaly'].values

    tp = np.sum((true_anomalies == True) & (predictions == True))
    fp = np.sum((true_anomalies == False) & (predictions == True))
    fn = np.sum((true_anomalies == True) & (predictions == False))
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn) 
    f1 = 2 * (precision * recall) / (precision + recall) 
    
    results.append({
        'n_points': len(df_series),
        'n_anomalies': int(true_anomalies.sum()),
        'predicted_anomalies': int(predictions.sum()),
        'precision': precision,
        'recall': recall,
        'f1': f1,
    })
    
    df_series['predicted'] = predictions
    all_predictions.append(df_series)
    
    print(f"Ряд {id} точек={len(df_series)} аномалий={int(true_anomalies.sum())} найдено={int(predictions.sum())} F1={f1:.3f}")

results_df = pd.DataFrame(results)
print(f"Precision {results_df['precision'].mean():.3f}")
print(f"Recall    {results_df['recall'].mean():.3f}")
print(f"F1        {results_df['f1'].mean():.3f}")
print(f"точек       {results_df['n_points'].sum()}")
print(f"аномалий    {results_df['n_anomalies'].sum()}")
print(f"найдено     {results_df['predicted_anomalies'].sum()}")