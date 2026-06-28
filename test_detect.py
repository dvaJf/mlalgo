import numpy as np
import pandas as pd
from gener import SineGenerator, SIGNAL_TYPES
from detect import detect
from detectai import detect_ml
from metrics import compute_metrics
import plotly.graph_objects as go
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
        noise = st.slider("Шум", 0.01, 1.0, 0.1, 0.01, format="%.2f")
        n_points = st.slider("Количество точек", 100, 5000, 500, 50)
        amplitude = st.slider("Амплитуда", 0.1, 5.0, 1.0, 0.1, format="%.1f")
        frequency = st.slider("Частота", 1, 10, 1, 1)
        col_s, col_e = st.columns(2)
        with col_s:
            start = st.number_input("Начало", value=0.0, step=1.0, format="%.1f")
        with col_e:
            end = st.number_input("Конец", value=35.0, step=1.0, format="%.1f")

        # Параметры аномалий
        st.markdown("### Параметры аномалий")
        anomaly_count = st.slider("Количество", 10, 200, 15, 1)
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
        ["Оба метода", "Статистический", "ML"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    generate_btn = st.button("Сгенерировать данные", width='stretch')


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

anomalies_stat, details_stat = detect(df, return_details=True) if show_stat else (None, None)
anomalies_ml, details_ml = detect_ml(df, return_details=True) if show_ml else (None, None)

def _build_chart_plotly(title, detected, label, method_type=None, details=None):
    """Строит интерактивный график временного ряда с отмеченными аномалиями."""
    fig = go.Figure()
    
    if method_type == "stat" and details is not None:
        fig.add_trace(go.Scatter(
            x=df['x'], y=details['lower'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=df['x'], y=details['upper'],
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(255, 255, 255, 0.1)',
            line=dict(width=0),
            name='Доверительный интервал',
            hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=df['x'], y=details['median'],
            mode='lines',
            line=dict(color='rgba(255, 255, 255, 0.3)', width=1, dash='dash'),
            name='Медиана'
        ))

    # Сигнал
    fig.add_trace(go.Scatter(
        x=df['x'], y=df['y'],
        mode='lines',
        name='Сигнал',
        line=dict(color='#2ecc71', width=1.5),
        opacity=0.7,
        hoverinfo='x+y'
    ))
    
    # Истинные аномалии
    if has_true_labels and df['is_anomaly'].any():
        fig.add_trace(go.Scatter(
            x=df.loc[df['is_anomaly'], 'x'], 
            y=df.loc[df['is_anomaly'], 'y'],
            mode='markers',
            name='Истинные аномалии',
            marker=dict(color='#e74c3c', size=5, symbol='circle'),
            opacity=0.9,
            hoverinfo='x+y'
        ))
        
    # Найденные алгоритмом
    if detected is not None and detected.any():
        if method_type == "ml" and details is not None:
            # Для ML отрисовываем красивый цветовой маппинг
            scores = details['scores']
            threshold = details['threshold']
            
            # Нормализация скоров для цвета
            norm_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
            
            custom_colorscale = [
                [0.0, 'rgba(255, 50, 50, 1.0)'],    # Аномалия - яркий красный
                [0.5, 'rgba(255, 165, 0, 0.8)'],    # Переходная зона - оранжевый
                [1.0, 'rgba(46, 204, 113, 0.05)']   # Норма - почти прозрачный зеленый
            ]
            
            fig.add_trace(go.Scatter(
                x=df['x'], 
                y=df['y'],
                mode='markers',
                name='Уровень уверенности',
                marker=dict(
                    size=5,
                    color=norm_scores,
                    colorscale=custom_colorscale,
                    showscale=True,
                    colorbar=dict(
                        title="Score", 
                        thickness=10, 
                        len=0.5, 
                        y=0.5,
                        tickfont=dict(color='#a0a0b0')
                    )
                ),
                hovertemplate='Уверенность: %{customdata:.2f}<extra></extra>',
                customdata=norm_scores
            ))
            
            # Рисуем обычные крестики для подтвержденных аномалий (как в stat методе)
            fig.add_trace(go.Scatter(
                x=df.loc[detected, 'x'], 
                y=df.loc[detected, 'y'],
                mode='markers',
                name=label,
                marker=dict(color='#3498db', size=10, symbol='x', line=dict(width=2, color='#3498db')),
                opacity=1.0,
                hoverinfo='x+y'
            ))
        else:
            fig.add_trace(go.Scatter(
                x=df.loc[detected, 'x'], 
                y=df.loc[detected, 'y'],
                mode='markers',
                name=label,
                marker=dict(color='#3498db', size=10, symbol='x', line=dict(width=2, color='#3498db')),
                opacity=1.0,
                hoverinfo='x+y'
            ))

    fig.update_xaxes(title_text="x", gridcolor='rgba(37, 37, 58, 0.5)', zerolinecolor='rgba(37, 37, 58, 0.5)')
    fig.update_yaxes(title_text="y", gridcolor='rgba(37, 37, 58, 0.5)', zerolinecolor='rgba(37, 37, 58, 0.5)')

    fig.update_layout(
        title=dict(text=title, font=dict(color="#ffffff", size=16)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#ffffff"),
        legend=dict(
            bgcolor='rgba(26, 26, 46, 0.8)',
            bordercolor='#35354a',
            borderwidth=1,
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=40, r=40, t=60, b=80),
        hovermode="x unified"
    )
    return fig


# Графики
if show_stat and show_ml:
    col1, col2 = st.columns(2)
elif show_stat:
    col1 = st.container()
else:
    col2 = st.container()

if show_stat:
    with col1:
        fig1 = _build_chart_plotly("Статистический метод (Z-score)", anomalies_stat, "Найдено", method_type="stat", details=details_stat)
        st.plotly_chart(fig1, width='stretch', config={'displayModeBar': False})

if show_ml:
    with col2:
        fig2 = _build_chart_plotly("Метод ML (Isolation Forest)", anomalies_ml, "Найдено", method_type="ml", details=details_ml)
        st.plotly_chart(fig2, width='stretch', config={'displayModeBar': False})


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

