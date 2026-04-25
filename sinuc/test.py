import numpy as np
import pandas as pd
from detect import detect

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

    # --- Расчет метрик классификации ---
    # TP: истинные аномалии, корректно найденные детектором
    tp = np.sum((true_anomalies == True) & (predictions == True))
    # FP: обычные точки, ошибочно помеченные как аномалии
    fp = np.sum((true_anomalies == False) & (predictions == True))
    # FN: аномалии, которые детектор пропустил
    fn = np.sum((true_anomalies == True) & (predictions == False))

    # Защита от деления на ноль (если TP+FP или TP+FN равны 0)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # F1-score — гармоническое среднее Precision и Recall
    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    # Сохранение результатов по текущему ряду
    results.append({
        'n_points': len(df_series),
        'n_anomalies': int(true_anomalies.sum()),
        'predicted_anomalies': int(predictions.sum()),
        'precision': precision,
        'recall': recall,
        'f1': f1,
    })

    # Промежуточный вывод для мониторинга процесса
    print(f"Ряд {id} точек={len(df_series)} аномалий={int(true_anomalies.sum())} "
          f"найдено={int(predictions.sum())} F1={f1:.3f}")

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