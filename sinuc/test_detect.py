import numpy as np
import pandas as pd
from gener import SineGenerator, SIGNAL_TYPES
from detect import detect
from detectai import detect_ml
from metrics import compute_metrics
import matplotlib.pyplot as plt
import streamlit as st

# Конфигурация страницы
st.set_page_config(
    page_title="Anomaly Detection — Sinuc",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Кастомные стили
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    }
    .main-title {
        text-align: center;
        font-size: 2.4rem;
        font-weight: 700;
        color: #e0e0e0;
        letter-spacing: 0.02em;
        margin-bottom: 0.2rem;
    }
    .main-subtitle {
        text-align: center;
        font-size: 1.05rem;
        color: #7a7a9e;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px 20px;
    }
    div[data-testid="stMetric"] label {
        color: #9e9ebb !important;
        font-size: 0.82rem !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #e8e8f0 !important;
        font-size: 1.8rem !important;
        font-weight: 600;
    }
    section[data-testid="stSidebar"] {
        background: rgba(15, 15, 26, 0.95) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #c0c0d8;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 1.2rem;
    }
    section[data-testid="stSidebar"] button[kind="secondary"] {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%) !important;
        color: #0f0f1a !important;
        border: none !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
    }
    .section-header {
        color: #c8c8e0;
        font-size: 1.15rem;
        font-weight: 600;
        border-bottom: 2px solid rgba(46, 204, 113, 0.3);
        padding-bottom: 8px;
        margin-bottom: 16px;
        margin-top: 8px;
    }
    .formula-hint {
        color: #6a6a8e;
        font-size: 0.8rem;
        line-height: 1.4;
    }
    .upload-info {
        color: #8a8aaa;
        font-size: 0.85rem;
        padding: 8px 12px;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
</style>
""", unsafe_allow_html=True)

# Заголовок
st.markdown('<div class="main-title">Anomaly Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">Поиск аномалий в числовых последовательностях</div>', unsafe_allow_html=True)

# Боковая панель
with st.sidebar:

    # Источник данных
    st.markdown("### Источник данных")
    data_source = st.radio(
        "Выберите источник",
        ["Генерация сигнала", "Загрузка CSV"],
        label_visibility="collapsed"
    )

    if data_source == "Генерация сигнала":
        # Тип сигнала
        st.markdown("### Тип сигнала")
        signal_label = st.selectbox(
            "Форма волны",
            list(SIGNAL_TYPES.keys()),
            label_visibility="collapsed"
        )
        signal_type = SIGNAL_TYPES[signal_label]

        # Поле для пользовательской формулы
        formula = None
        if signal_type == "custom":
            formula = st.text_input(
                "Формула",
                value="x**2+x",
                help="Переменные: x, a (амплитуда), f (частота). Функции: sin, cos, tan, exp, log, sqrt, abs, pi"
            )


        # Параметры сигнала
        st.markdown("### Параметры сигнала")
        noise = st.slider("Шум (std)", 0.01, 1.0, 0.1, 0.01, format="%.2f")
        n_points = st.slider("Количество точек", 100, 5000, 500, 50)
        amplitude = st.slider("Амплитуда", 0.1, 5.0, 1.0, 0.1, format="%.1f")
        frequency = st.slider("Частота", 1, 10, 1, 1)
        col_s, col_e = st.columns(2)
        with col_s:
            start = st.number_input("Начало", value=0.0, step=1.0, format="%.1f")
        with col_e:
            end = st.number_input("Конец", value=25, step=1.0, format="%.1f")

        # Параметры аномалий
        st.markdown("### Параметры аномалий")
        anomaly_count = st.slider("Количество", 1, 200, 15, 1)
        anomaly_scale = st.slider("Амплитуда", 0.5, 10.0, 3.0, 0.1, format="%.1f",
                                  key="anomaly_scale_slider")

    else:
        # Загрузка CSV
        st.markdown("### Загрузка файла")
        uploaded_file = st.file_uploader(
            "CSV файл",
            type=["csv"],
            label_visibility="collapsed"
        )


    # Метод детекции
    st.markdown("### Метод детекции")
    method = st.radio(
        "Метод",
        ["Оба метода", "Статистический (Z-score)", "ML (Isolation Forest)"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    generate_btn = st.button("Сгенерировать данные", use_container_width=True)


# Подготовка данных
df = None
has_true_labels = False

if data_source == "Генерация сигнала":
    if generate_btn or 'sg_df' not in st.session_state:
        try:
            sg = SineGenerator(
                noise=noise, n_points=n_points, start=start, end=end,
                amplitude=amplitude, frequency=frequency,
                anomaly_count=anomaly_count, anomaly_scale=anomaly_scale,
                signal_type=signal_type, formula=formula
            )
            sg.generate()
            st.session_state['sg_df'] = sg.df.copy()
        except ValueError as e:
            st.error(f"Ошибка генерации: {e}")
            st.stop()

    df = st.session_state['sg_df']
    has_true_labels = True

else:
    # Загрузка CSV
    if uploaded_file is not None:
        try:
            raw = pd.read_csv(uploaded_file)

            if 'y' not in raw.columns:
                st.error("В файле отсутствует колонка 'y'. Убедитесь, что CSV содержит колонку с числовыми значениями под названием 'y'.")
                st.stop()

            df = pd.DataFrame()
            df['y'] = pd.to_numeric(raw['y'], errors='coerce')

            if 'x' in raw.columns:
                df['x'] = pd.to_numeric(raw['x'], errors='coerce')
            else:
                df['x'] = np.arange(len(df), dtype=float)

            if 'is_anomaly' in raw.columns:
                df['is_anomaly'] = raw['is_anomaly'].astype(bool)
                has_true_labels = True
            else:
                df['is_anomaly'] = False
                has_true_labels = False

            # Удаление строк с NaN
            df = df.dropna().reset_index(drop=True)

            st.session_state['sg_df'] = df.copy()

        except Exception as e:
            st.error(f"Ошибка чтения CSV: {e}")
            st.stop()
    elif 'sg_df' in st.session_state:
        df = st.session_state['sg_df']
        has_true_labels = 'is_anomaly' in df.columns and df['is_anomaly'].any()
    else:
        st.info("Загрузите CSV файл или переключитесь на генерацию сигнала.")
        st.stop()


# Детекция
show_stat = method in ["Оба метода", "Статистический (Z-score)"]
show_ml = method in ["Оба метода", "ML (Isolation Forest)"]

anomalies_stat = detect(df) if show_stat else None
anomalies_ml = detect_ml(df) if show_ml else None

# Стиль matplotlib
plt.rcParams.update({
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "text.color": "#a0a0c0",
    "axes.labelcolor": "#a0a0c0",
    "xtick.color": "#707090",
    "ytick.color": "#707090",
    "axes.edgecolor": "#35354a",
    "grid.color": "#25253a",
    "grid.alpha": 0.5,
    "grid.linestyle": "--",
    "font.size": 10,
    "legend.facecolor": "#1a1a2e",
    "legend.edgecolor": "#35354a",
})


def _build_chart(ax, title, detected, label):
    """Строит график временного ряда с отмеченными аномалиями."""
    ax.set_title(title, color="#c8c8e0", fontsize=12, fontweight=600, pad=12)
    ax.plot(df['x'], df['y'], color='#2ecc71', alpha=0.5, linewidth=1.2, label="Сигнал")

    # Истинные аномалии (если есть разметка)
    if has_true_labels and df['is_anomaly'].any():
        ax.scatter(
            df.loc[df['is_anomaly'], 'x'], df.loc[df['is_anomaly'], 'y'],
            color='#e74c3c', zorder=4, label='Истинные аномалии', s=40, alpha=0.85, edgecolors='none'
        )

    # Найденные алгоритмом
    ax.scatter(
        df.loc[detected, 'x'], df.loc[detected, 'y'],
        color='#3498db', zorder=5, label=label, marker='x', s=60, linewidth=2
    )
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


# Графики
if show_stat and show_ml:
    col1, col2 = st.columns(2)
elif show_stat:
    col1 = st.container()
else:
    col2 = st.container()

if show_stat:
    with col1:
        fig1, ax1 = plt.subplots(figsize=(10, 4.5))
        _build_chart(ax1, "Статистический метод (Z-score)", anomalies_stat, "Найдено (Z-score)")
        st.pyplot(fig1, transparent=True)
        plt.close(fig1)

if show_ml:
    with col2:
        fig2, ax2 = plt.subplots(figsize=(10, 4.5))
        _build_chart(ax2, "Isolation Forest", anomalies_ml, "Найдено (ML)")
        st.pyplot(fig2, transparent=True)
        plt.close(fig2)


# Метрики качества
st.markdown('<div class="section-header">Метрики качества</div>', unsafe_allow_html=True)

if has_true_labels:
    true_anomalies = df['is_anomaly']

    if show_stat and show_ml:
        m_stat = compute_metrics(true_anomalies, anomalies_stat)
        m_ml = compute_metrics(true_anomalies, anomalies_ml)
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Stat Precision", f"{m_stat['precision']:.4f}")
        c2.metric("Stat Recall", f"{m_stat['recall']:.4f}")
        c3.metric("Stat F1", f"{m_stat['f1']:.4f}")
        c4.metric("ML Precision", f"{m_ml['precision']:.4f}")
        c5.metric("ML Recall", f"{m_ml['recall']:.4f}")
        c6.metric("ML F1", f"{m_ml['f1']:.4f}")
    elif show_stat:
        m_stat = compute_metrics(true_anomalies, anomalies_stat)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Precision", f"{m_stat['precision']:.4f}")
        c2.metric("Recall", f"{m_stat['recall']:.4f}")
        c3.metric("F1-score", f"{m_stat['f1']:.4f}")
        c4.metric("Найдено / Истинных", f"{int(anomalies_stat.sum())} / {int(true_anomalies.sum())}")
    elif show_ml:
        m_ml = compute_metrics(true_anomalies, anomalies_ml)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Precision", f"{m_ml['precision']:.4f}")
        c2.metric("Recall", f"{m_ml['recall']:.4f}")
        c3.metric("F1-score", f"{m_ml['f1']:.4f}")
        c4.metric("Найдено / Истинных", f"{int(anomalies_ml.sum())} / {int(true_anomalies.sum())}")
else:
    # Нет истинных меток — показываем только количество найденных
    if show_stat and show_ml:
        c1, c2, c3 = st.columns(3)
        c1.metric("Точек в ряде", len(df))
        c2.metric("Stat: найдено аномалий", int(anomalies_stat.sum()))
        c3.metric("ML: найдено аномалий", int(anomalies_ml.sum()))
    elif show_stat:
        c1, c2 = st.columns(2)
        c1.metric("Точек в ряде", len(df))
        c2.metric("Найдено аномалий", int(anomalies_stat.sum()))
    elif show_ml:
        c1, c2 = st.columns(2)
        c1.metric("Точек в ряде", len(df))
        c2.metric("Найдено аномалий", int(anomalies_ml.sum()))

    st.caption("Метрики Precision / Recall / F1 недоступны: в данных нет истинных меток аномалий (колонка is_anomaly).")

