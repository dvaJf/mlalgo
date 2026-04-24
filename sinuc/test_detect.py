import numpy as np
from gener import SineGenerator
from detect import detect
from detectai import detect_ml
import matplotlib.pyplot as plt
import streamlit as st
st.set_page_config(page_title="Anomaly Detection", layout="wide")
st.title("Выявление Аномалий", text_alignment="center")
side_col, main_col = st.columns([1, 5], border=True, gap="small")
with side_col:
    noise = st.number_input("Шум", min_value=0.01, value=0.1, step=0.01, format="%.2f")
    n_points = st.number_input("n-points", min_value=1, value=500, step=1)
    start = st.number_input("Начало", min_value=0, value=0, step=1)
    end = st.number_input("Конец", min_value=1.0, value=8*np.pi, step=0.01, format="%.2f")
    amplitude = st.number_input("Амплитуда", min_value=0.1, value=1.0, step=0.1, format="%.2f")
    frequency = st.number_input("Частота", min_value=1, value=1, step=1)
    anomaly_count = st.number_input("Кол-во точек", min_value=1, value=15, step=1)
    anomaly_scale = st.number_input("Размер точек", min_value=1.0, value=2.0, step=0.1, format="%.2f")

sg = SineGenerator(noise=noise,n_points=n_points, start=start, end=end, amplitude=amplitude, frequency=frequency, anomaly_count=anomaly_count, anomaly_scale=anomaly_scale)
sg.generate()  # генерация данных, синусоиды

anomalies_stat = detect(sg.df)  # алгоритм, массив точек потенц аномалий
anomalies_ml = detect_ml(sg.df)  # алгоритм мл массив точек потенц аномалий

true_anomalies = sg.df['is_anomaly']  # массив аномалий, тру аномалии




# Создаем две колонки одинаковой ширины
with main_col:
    st.subheader("Сравнение поиска аномалий", text_alignment='center', divider="green")
    col1, col2 = st.columns(2)
    plt.style.use('seaborn-v0_8-muted')  # Современный чистый стиль
    plt.rcParams.update({
        "text.color": "#888888",  # Нейтральный цвет текста
        "axes.labelcolor": "#888888",  # Цвет названий осей
        "xtick.color": "#888888",  # Цвет делений на оси X
        "ytick.color": "#888888",  # Цвет делений на оси Y
        "axes.edgecolor": "#444444",  # Цвет рамки графика
        "grid.color": "#444444",  # Цвет сетки
        "grid.alpha": 0.2  # Прозрачность сетки
    })
    with col1:
        st.markdown("**Статистический метод (Алгоритм)**")
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(sg.df['x'], sg.df['y'], color='#2ecc71', alpha=0.6, linewidth=1.5)
        ax1.scatter(sg.df.loc[sg.df['is_anomaly'], 'x'],
               sg.df.loc[sg.df['is_anomaly'], 'y'],
               color='#780000', zorder=4, label='Аномалии', s=50)
        ax1.scatter(sg.df.loc[anomalies_stat, 'x'],
               sg.df.loc[anomalies_stat, 'y'],
               color='#1e90ff', zorder=5, label="Алгоритм", marker='x', s=70, linewidth=2)
        ax1.legend(facecolor='none', edgecolor='#2ecc71', fontsize='small', loc='upper right')
        ax1.grid(True, linestyle='--')
        st.pyplot(fig1, transparent=True)
        col1.metric("Аномалий выявлено", len(sg.df.loc[anomalies_stat, 'x']), delta =len(sg.df.loc[anomalies_stat, 'x']) - len(sg.df.loc[anomalies_ml, 'x']))

    with col2:
        st.markdown("**Метод машинного обучения (ML)**")
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.plot(sg.df['x'], sg.df['y'], color='#2ecc71', alpha=0.6, linewidth=1.5)
        ax2.scatter(sg.df.loc[sg.df['is_anomaly'], 'x'],
               sg.df.loc[sg.df['is_anomaly'], 'y'],
               color='#780000', zorder=4, label='Аномалии', s=50)
        ax2.scatter(sg.df.loc[anomalies_ml, 'x'], sg.df.loc[anomalies_ml, 'y'],
               color='#1e90ff', zorder=5, label="ML", marker='x', s=70, linewidth=2)
        ax2.legend(facecolor='none', edgecolor='#2ecc71', fontsize='small', loc='upper right')
        ax2.grid(True, linestyle='--')
        st.pyplot(fig2, transparent=True)
        col2.metric("Аномалий выявлено", len(sg.df.loc[anomalies_ml, 'x']), delta=len(sg.df.loc[anomalies_ml, 'x'])-len(sg.df.loc[anomalies_stat, 'x']))

st.divider()

name, anomalies = ('ML', anomalies_ml)
detected_anomalies = anomalies

tp = np.sum((true_anomalies == True) & (detected_anomalies == True))  # true positive позитивные/все срабатывания
fp = np.sum((true_anomalies == False) & (detected_anomalies == True))  # false positive
fn = np.sum((true_anomalies == True) & (detected_anomalies == False))  # false negative

precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = 2 * (precision * recall) / (precision + recall)
c1, c2, c3 = st.columns(3)
c1.metric("Precision", format(precision, ".4f"))
c2.metric("recall", format(recall, ".4f"))
c3.metric("f1_score", format(f1_score, ".4f"))


name, anomalies = ('default', anomalies_stat)
detected_anomalies = anomalies

tp = np.sum((true_anomalies == True) & (detected_anomalies == True))  # true positive позитивные/все срабатывания
fp = np.sum((true_anomalies == False) & (detected_anomalies == True))  # false positive
fn = np.sum((true_anomalies == True) & (detected_anomalies == False))  # false negative

precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = 2 * (precision * recall) / (precision + recall)

c1, c2, c3 = st.columns(3)
c1.metric("Precision", format(precision, ".4f"))
c2.metric("recall", format(recall, ".4f"))
c3.metric("f1_score", format(f1_score, ".4f"))