import numpy as np
import pandas as pd
from detect import detect
from metrics import compute_metrics

# Загрузка датасета с множественными временными рядами
df = pd.read_csv('test_data.csv')

# Уникальные идентификаторы рядов
ids = df['id'].unique()

results = []

# Последовательная обработка каждого ряда
for id in ids:
    # Выделение одного ряда по его id
    df_series = df[df['id'] == id].copy()

    # Запуск статистического детектора аномалий
    predictions = detect(df_series)

    # Истинные метки аномалий из генератора
    true_anomalies = df_series['is_anomaly'].values

    # Расчёт метрик классификации
    m = compute_metrics(true_anomalies, predictions)

    # Сохранение результатов по текущему ряду
    results.append({
        'n_points': len(df_series),
        'n_anomalies': int(true_anomalies.sum()),
        'predicted_anomalies': int(predictions.sum()),
        'precision': m['precision'],
        'recall': m['recall'],
        'f1': m['f1'],
    })

    # Промежуточный вывод для мониторинга процесса
    print(f"Ряд {id} точек={len(df_series)} аномалий={int(true_anomalies.sum())} "
          f"найдено={int(predictions.sum())} F1={m['f1']:.3f}")

# Агрегация результатов по всем рядам
results_df = pd.DataFrame(results)

# Вывод усредненных метрик по датасету
print(f"Precision {results_df['precision'].mean():.3f}")
print(f"Recall    {results_df['recall'].mean():.3f}")
print(f"F1        {results_df['f1'].mean():.3f}")

# Вывод сводных абсолютных величин
print(f"точек       {results_df['n_points'].sum()}")
print(f"аномалий    {results_df['n_anomalies'].sum()}")
print(f"найдено     {results_df['predicted_anomalies'].sum()}")