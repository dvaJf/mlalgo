import numpy as np
import pandas as pd
from gener import SineGenerator

# Количество временных рядов в датасете
n = 100
all_data = []

for id in range(n):
    # Инициализация генератора для одного ряда
    sg = SineGenerator(
        n_points=3000,        # Длина каждого ряда
        noise=0.1,            # Уровень шума
        anomaly_count=150,    # Количество аномалий в ряду (~5% от длины)
        anomaly_scale=2.0,    # Амплитуда скачка аномалии
        amplitude=1.0,        # Амплитуда синусоиды
        frequency=1.0,        # Частота синусоиды
        start=0,              # Начало интервала
        end=64 * np.pi        # Конец интервала (32 полных периода)
    )

    # Генерация ряда
    sg.generate()

    # Копирование DataFrame и добавление идентификатора ряда
    df = sg.df.copy()
    df['id'] = id

    # Перестановка колонок в удобный порядок
    df = df[['id', 'x', 'y', 'is_anomaly']]
    all_data.append(df)

# Объединение всех рядов в единый DataFrame
full_data = pd.concat(all_data, ignore_index=True)

# Сохранение датасета в CSV (без индекса)
full_data.to_csv('test_data.csv', index=False)

# Вывод сводной статистики
print(f"точек {len(full_data)}")
print(f"аномалий {full_data['is_anomaly'].sum()}")
print(f"рядов {n}")