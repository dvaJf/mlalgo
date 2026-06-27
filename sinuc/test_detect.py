import numpy as np
from gener import SineGenerator
from detect import detect
from detectai import detect_ml
from metrics import compute_metrics
import matplotlib.pyplot as plt
import streamlit as st

# ─── Конфигурация страницы ───
st.set_page_config(
    page_title="Anomaly Detection — Sinuc",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Кастомные стили (тёмная тема, современный вид) ───
st.markdown("""
<style>
    /* Основной фон и типографика */
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    }

    /* Заголовок страницы */
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

    /* Карточки метрик */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px 20px;
        backdrop-filter: blur(10px);
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

    /* Боковая панель */
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

    /* Кнопка генерации */
    section[data-testid="stSidebar"] button[kind="secondary"] {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%) !important;
        color: #0f0f1a !important;
        border: none !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        transition: all 0.2s ease;
    }
    section[data-testid="stSidebar"] button[kind="secondary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(46, 204, 113, 0.3);
    }

    /* Заголовки секций */
    .section-header {
        color: #c8c8e0;
        font-size: 1.15rem;
        font-weight: 600;
        border-bottom: 2px solid rgba(46, 204, 113, 0.3);
        padding-bottom: 8px;
        margin-bottom: 16px;
        margin-top: 8px;
    }

    /* Названия методов на графиках */
    .method-label {
        color: #a0a0c0;
        font-size: 0.95rem;
        font-weight: 500;
        margin-bottom: 8px;
    }

    /* Expander */
    details {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 10px !important;
    }
    details summary {
        color: #b0b0cc !important;
        font-weight: 500;
    }

    /* Скрыть стандартную рамку Streamlit */
    .stColumn > div {
        padding: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ─── Заголовок ───
st.markdown('<div class="main-title">Anomaly Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">Поиск аномалий в синтетических временных рядах</div>', unsafe_allow_html=True)

# ─── Боковая панель ───
with st.sidebar:
    st.markdown("### Параметры сигнала")
    noise = st.slider("Шум (std)", min_value=0.01, max_value=1.0, value=0.1, step=0.01, format="%.2f")
    n_points = st.slider("Количество точек", min_value=100, max_value=5000, value=500, step=50)
    amplitude = st.slider("Амплитуда", min_value=0.1, max_value=5.0, value=1.0, step=0.1, format="%.1f")
    frequency = st.slider("Частота", min_value=1, max_value=10, value=1, step=1)
    col_s, col_e = st.columns(2)
    with col_s:
        start = st.number_input("Начало", min_value=0.0, value=0.0, step=1.0, format="%.1f")
    with col_e:
        end = st.number_input("Конец", min_value=1.0, value=8*np.pi, step=1.0, format="%.1f")

    st.markdown("### Параметры аномалий")
    anomaly_count = st.slider("Количество аномалий", min_value=1, max_value=200, value=15, step=1)
    anomaly_scale = st.slider("Амплитуда аномалии", min_value=0.5, max_value=10.0, value=2.0, step=0.1, format="%.1f")

    st.markdown("### Метод детекции")
    method = st.radio(
        "Выберите метод",
        ["Оба метода", "Статистический (Z-score)", "ML (Isolation Forest)"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    generate_btn = st.button("Сгенерировать данные", use_container_width=True)

# ─── Управление данными через session_state ───
if generate_btn or 'sg_df' not in st.session_state:
    sg = SineGenerator(
        noise=noise, n_points=n_points, start=start, end=end,
        amplitude=amplitude, frequency=frequency,
        anomaly_count=anomaly_count, anomaly_scale=anomaly_scale
    )
    sg.generate()
    st.session_state['sg_df'] = sg.df.copy()

df = st.session_state['sg_df']

# ─── Детекция ───
show_stat = method in ["Оба метода", "Статистический (Z-score)"]
show_ml = method in ["Оба метода", "ML (Isolation Forest)"]

anomalies_stat = detect(df) if show_stat else None
anomalies_ml = detect_ml(df) if show_ml else None
true_anomalies = df['is_anomaly']

# ─── Стиль matplotlib ───
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
    # Основной сигнал
    ax.plot(df['x'], df['y'], color='#2ecc71', alpha=0.5, linewidth=1.2, label="Сигнал")
    # Истинные аномалии (красные точки)
    ax.scatter(
        df.loc[df['is_anomaly'], 'x'], df.loc[df['is_anomaly'], 'y'],
        color='#e74c3c', zorder=4, label='Истинные аномалии', s=40, alpha=0.85, edgecolors='none'
    )
    # Найденные алгоритмом (синие крестики)
    ax.scatter(
        df.loc[detected, 'x'], df.loc[detected, 'y'],
        color='#3498db', zorder=5, label=label, marker='x', s=60, linewidth=2
    )
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


# ─── Графики ───
if show_stat and show_ml:
    col1, col2 = st.columns(2)
elif show_stat:
    col1 = st.container()
else:
    col2 = st.container()

if show_stat:
    with col1:
        fig1, ax1 = plt.subplots(figsize=(10, 4.5))
        _build_chart(ax1, "Статистический метод (Z-score + MAD)", anomalies_stat, "Найдено (Z-score)")
        st.pyplot(fig1, transparent=True)
        plt.close(fig1)

if show_ml:
    with col2:
        fig2, ax2 = plt.subplots(figsize=(10, 4.5))
        _build_chart(ax2, "Isolation Forest", anomalies_ml, "Найдено (ML)")
        st.pyplot(fig2, transparent=True)
        plt.close(fig2)

# ─── Метрики качества ───
st.markdown('<div class="section-header">Метрики качества</div>', unsafe_allow_html=True)

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

# ─── Описание методов ───
st.markdown("---")
st.markdown('<div class="section-header">Описание алгоритмов</div>', unsafe_allow_html=True)

with st.expander("Статистический метод (Z-score + MAD)"):
    st.markdown("""
Детектор использует два статистических подхода одновременно:

**Z-score** — для каждой точки берётся окно из соседних значений, вычисляется среднее
и стандартное отклонение. Если точка отклоняется больше чем на `n_sigma` стандартных
отклонений, она считается аномалией.

**MAD (Median Absolute Deviation)** — аналогичный подход, но вместо среднего используется
медиана, а вместо стандартного отклонения — медианное абсолютное отклонение.
Медиана устойчива к выбросам: даже если в окно попало несколько аномалий,
они не сдвинут центр и не раздуют разброс.

Точка считается аномалией, если хотя бы один из двух методов её обнаружил.
    """)

with st.expander("ML-метод (Isolation Forest)"):
    st.markdown("""
**Isolation Forest** — алгоритм обучения без учителя из библиотеки scikit-learn.

Для каждой точки вычисляются 11 инженерных признаков:
- Значение, первая и вторая производные
- Остатки после сглаживания с двумя размерами окна
- Локальное стандартное отклонение и Z-score
- Размах в локальном окне (max - min)
- Отклонения от глобального среднего и медианы

Алгоритм строит 500 случайных деревьев решений. Аномалии изолируются
за меньшее число разбиений, чем нормальные точки. Порог отсечения
выбирается автоматически: алгоритм ищет точку максимального скачка
в отсортированном распределении скоров аномальности.
    """)