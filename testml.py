import numpy as np
import pandas as pd
from detectai import detect_ml
from metrics import compute_metrics

# Датасет
df = pd.read_csv('test_data.csv')

# ID
ids = df['id'].unique()

results = []

# Обработка
for id in ids:
    df_series = df[df['id'] == id].copy()

    # ML
    predictions = detect_ml(df_series)

    true_anomalies = df_series['is_anomaly'].values

    # Метрики
    m = compute_metrics(true_anomalies, predictions)

    results.append({
        'n_points': len(df_series),
        'n_anomalies': int(true_anomalies.sum()),
        'predicted_anomalies': int(predictions.sum()),
        'precision': m['precision'],
        'recall': m['recall'],
        'f1': m['f1'],
    })

    print(f"Ряд {id} точек={len(df_series)} аномалий={int(true_anomalies.sum())} "
          f"найдено={int(predictions.sum())} F1={m['f1']:.3f}")

# Итог
results_df = pd.DataFrame(results)

print(f"Precision {results_df['precision'].mean():.3f}")
print(f"Recall    {results_df['recall'].mean():.3f}")
print(f"F1        {results_df['f1'].mean():.3f}")

print(f"точек       {results_df['n_points'].sum()}")
print(f"аномалий    {results_df['n_anomalies'].sum()}")
print(f"найдено     {results_df['predicted_anomalies'].sum()}")