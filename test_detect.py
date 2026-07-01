import numpy as np
import pandas as pd
from gener import SineGenerator, SIGNAL_TYPES
from detect import detect
from detectai import detect_ml
from metrics import compute_metrics
import plotly.graph_objects as go
import streamlit as st

# Конфиг
st.set_page_config(
    page_title="Anomaly Detection — Sinuc",
    layout="wide",
    initial_sidebar_state="expanded"
)

is_light_theme = st.sidebar.toggle(" Светлая тема", value=False)

# Стили
if is_light_theme:
    st.markdown("""
    <style>
        .stApp { background: linear-gradient(135deg, #ffffff 0%, #f0f4f8 100%); }
        p, span, label, h1, h2, h3, h4, h5, h6, li { color: #0f172a !important; }
        .main-title { text-align: center; font-size: 2.4rem; font-weight: 700; color: #1e3a8a !important; letter-spacing: 0.02em; margin-bottom: 0.2rem; }
        .main-subtitle { text-align: center; font-size: 1.05rem; color: #3b82f6 !important; margin-bottom: 2rem; font-weight: 400; }
        div[data-testid="stMetric"] { background: #ffffff; border: 1px solid rgba(59, 130, 246, 0.2); border-radius: 12px; padding: 16px 20px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); }
        div[data-testid="stMetric"] label, div[data-testid="stMetric"] label p { color: #64748b !important; font-size: 0.82rem !important; text-transform: uppercase; letter-spacing: 0.06em; }
        div[data-testid="stMetric"] div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] div { color: #1e293b !important; font-size: 1.8rem !important; font-weight: 600; }
        section[data-testid="stSidebar"] { background: #f8fafc !important; border-right: 1px solid rgba(0, 0, 0, 0.05); }
        section[data-testid="stSidebar"] .stMarkdown h3, section[data-testid="stSidebar"] .stMarkdown h3 p { color: #334155 !important; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 1.2rem; }
        button[kind="secondary"] { background: linear-gradient(135deg, #f97316 0%, #ea580c 100%) !important; color: #ffffff !important; border: none !important; font-weight: 600 !important; border-radius: 8px !important; }
        button[kind="secondary"] p, button[kind="secondary"] span { color: #ffffff !important; }
        .section-header, .section-header p { color: #1e3a8a !important; font-size: 1.15rem; font-weight: 600; border-bottom: 2px solid rgba(249, 115, 22, 0.4); padding-bottom: 8px; margin-bottom: 16px; margin-top: 8px; }
        .formula-hint, .formula-hint p { color: #64748b !important; font-size: 0.8rem; line-height: 1.4; }
        .upload-info, .upload-info p { color: #64748b !important; font-size: 0.85rem; padding: 8px 12px; background: rgba(59, 130, 246, 0.05); border-radius: 8px; border: 1px solid rgba(59, 130, 246, 0.1); }
        input, select, textarea, div[data-baseweb="select"] > div, div[data-baseweb="base-input"] { background-color: #ffffff !important; color: #0f172a !important; border-color: #cbd5e1 !important; }
        div[data-baseweb="radio"] div { color: #0f172a !important; }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
        .stApp { background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%); }
        .main-title { text-align: center; font-size: 2.4rem; font-weight: 700; color: #e0e0e0; letter-spacing: 0.02em; margin-bottom: 0.2rem; }
        .main-subtitle { text-align: center; font-size: 1.05rem; color: #7a7a9e; margin-bottom: 2rem; font-weight: 400; }
        div[data-testid="stMetric"] { background: rgba(255, 255, 255, 0.04); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 16px 20px; box-shadow: none; }
        div[data-testid="stMetric"] label { color: #9e9ebb !important; font-size: 0.82rem !important; text-transform: uppercase; letter-spacing: 0.06em; }
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #e8e8f0 !important; font-size: 1.8rem !important; font-weight: 600; }
        section[data-testid="stSidebar"] { background: rgba(15, 15, 26, 0.95) !important; border-right: 1px solid rgba(255, 255, 255, 0.06); }
        section[data-testid="stSidebar"] .stMarkdown h3 { color: #c0c0d8; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 1.2rem; }
        section[data-testid="stSidebar"] button[kind="secondary"] { background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%) !important; color: #0f0f1a !important; border: none !important; font-weight: 600 !important; border-radius: 8px !important; }
        .section-header { color: #c8c8e0; font-size: 1.15rem; font-weight: 600; border-bottom: 2px solid rgba(46, 204, 113, 0.3); padding-bottom: 8px; margin-bottom: 16px; margin-top: 8px; }
        .formula-hint { color: #6a6a8e; font-size: 0.8rem; line-height: 1.4; }
        .upload-info { color: #8a8aaa; font-size: 0.85rem; padding: 8px 12px; background: rgba(255, 255, 255, 0.03); border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.06); }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="main-title">Anomaly Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">Поиск аномалий в числовых последовательностях</div>', unsafe_allow_html=True)

# Меню
with st.sidebar:

    data_source = st.radio(
        "Выберите источник",
        ["Генерация сигнала", "Загрузка CSV"],
        )

    if data_source == "Генерация сигнала":
        st.markdown("### Тип сигнала")
        signal_label = st.selectbox(
            "Форма волны",
            list(SIGNAL_TYPES.keys()),
            label_visibility="collapsed"
        )
        signal_type = SIGNAL_TYPES[signal_label]

        # Формула
        formula = None
        if signal_type == "custom":
            formula = st.text_input(
                "Формула",
                value="x**2+x",
                help="Переменные: x, a (амплитуда), f (частота). Функции: sin, cos, tan, exp, log, sqrt, abs, pi"
            )


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

        st.markdown("### Параметры аномалий")
        anomaly_count = st.slider("Количество", 10, 200, 15, 1)
        anomaly_scale = st.slider("Амплитуда", 0.5, 10.0, 3.0, 0.1, format="%.1f",
                                  key="anomaly_scale_slider")

    else:
        st.markdown("### Загрузка файла")
        uploaded_file = st.file_uploader(
            "CSV файл",
            type=["csv"],
            label_visibility="collapsed"
        )


    st.markdown("### Метод детекции")
    method = st.radio(
        "Метод",
        ["Оба метода", "Статистический", "ML"],
        label_visibility="collapsed"
    )

    show_ci = st.toggle("Доп информация", value=False,
                        help="Показать доверительный интервал и уровень уверенности")

    st.markdown("---")
    generate_btn = st.button("Сгенерировать данные", width='stretch')


# Данные
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
show_stat = method in ["Оба метода", "Статистический"]
show_ml = method in ["Оба метода", "ML"]

anomalies_stat, details_stat = detect(df, return_details=True) if show_stat else (None, None)
anomalies_ml, details_ml = detect_ml(df, return_details=True) if show_ml else (None, None)

def _build_chart_plotly(title, detected, label, method_type=None, details=None, show_ci=False):
    """График."""
    fig = go.Figure()
    
    # Цвета
    if is_light_theme:
        c_ci = 'rgba(59, 130, 246, 0.1)'
        c_median = 'rgba(30, 58, 138, 0.3)'
        c_signal = '#3b82f6'
        c_true = '#f97316'
        c_ml_scale = [
            [0.0, 'rgba(249, 115, 22, 1.0)'],
            [0.5, 'rgba(251, 146, 60, 0.8)'],
            [1.0, 'rgba(59, 130, 246, 0.1)']
        ]
        c_ml_tick = '#64748b'
        c_detected = '#f97316'
        c_detected_other = '#3498db'
        c_grid = 'rgba(226, 232, 240, 1.0)'
        c_title = '#1e3a8a'
        c_font = '#334155'
        c_legend_bg = 'rgba(255, 255, 255, 0.8)'
        c_legend_border = '#e2e8f0'
    else:
        c_ci = 'rgba(255, 255, 255, 0.1)'
        c_median = 'rgba(255, 255, 255, 0.3)'
        c_signal = '#2ecc71'
        c_true = '#e74c3c'
        c_ml_scale = [
            [0.0, 'rgba(255, 50, 50, 1.0)'],
            [0.5, 'rgba(255, 165, 0, 0.8)'],
            [1.0, 'rgba(46, 204, 113, 0.05)']
        ]
        c_ml_tick = '#a0a0b0'
        c_detected = '#3498db'
        c_detected_other = '#3498db'
        c_grid = 'rgba(37, 37, 58, 0.5)'
        c_title = '#ffffff'
        c_font = '#ffffff'
        c_legend_bg = 'rgba(26, 26, 46, 0.8)'
        c_legend_border = '#35354a'

    if method_type == "stat" and details is not None and show_ci:
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
            fillcolor=c_ci,
            line=dict(width=0),
            name='Доверительный интервал',
            hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=df['x'], y=details['median'],
            mode='lines',
            line=dict(color=c_median, width=1, dash='dash'),
            name='Медиана'
        ))

    fig.add_trace(go.Scatter(
        x=df['x'], y=df['y'],
        mode='lines',
        name='Сигнал',
        line=dict(color=c_signal, width=1.5),
        opacity=0.7,
        hoverinfo='x+y'
    ))
    
    if has_true_labels and df['is_anomaly'].any():
        fig.add_trace(go.Scatter(
            x=df.loc[df['is_anomaly'], 'x'], 
            y=df.loc[df['is_anomaly'], 'y'],
            mode='markers',
            name='Истинные аномалии',
            marker=dict(color=c_true, size=5, symbol='circle'),
            opacity=0.9,
            hoverinfo='x+y'
        ))
        
    if detected is not None and detected.any():
        if method_type == "ml" and details is not None:
            scores = details['scores']
            threshold = details['threshold']
            
            if show_ci:
                # Нормализация скоров для цвета
                norm_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
                
                fig.add_trace(go.Scatter(
                    x=df['x'], 
                    y=df['y'],
                    mode='markers',
                    name='Уровень уверенности',
                    marker=dict(
                        size=5,
                        color=norm_scores,
                        colorscale=c_ml_scale,
                        showscale=True,
                        colorbar=dict(
                            title="Score", 
                            thickness=10, 
                            len=0.5, 
                            y=0.5,
                            tickfont=dict(color=c_ml_tick)
                        )
                    ),
                    hovertemplate='Уверенность: %{customdata:.2f}<extra></extra>',
                    customdata=norm_scores
                ))
            
            # Рисуем обычные крестики для подтвержденных аномалий
            fig.add_trace(go.Scatter(
                x=df.loc[detected, 'x'], 
                y=df.loc[detected, 'y'],
                mode='markers',
                name=label,
                marker=dict(color=c_detected, size=10, symbol='x', line=dict(width=2, color=c_detected)),
                opacity=1.0,
                hoverinfo='x+y'
            ))
        else:
            fig.add_trace(go.Scatter(
                x=df.loc[detected, 'x'], 
                y=df.loc[detected, 'y'],
                mode='markers',
                name=label,
                marker=dict(color=c_detected_other, size=10, symbol='x', line=dict(width=2, color=c_detected_other)),
                opacity=1.0,
                hoverinfo='x+y'
            ))

    fig.update_xaxes(title_text="x", gridcolor=c_grid, zerolinecolor=c_grid)
    fig.update_yaxes(title_text="y", gridcolor=c_grid, zerolinecolor=c_grid)

    fig.update_layout(
        title=dict(text=title, font=dict(color=c_title, size=16)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=c_font),
        legend=dict(
            font=dict(color=c_font),
            bgcolor=c_legend_bg,
            bordercolor=c_legend_border,
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
        fig1 = _build_chart_plotly("Статистический метод (Z-score)", anomalies_stat, "Найдено", method_type="stat", details=details_stat, show_ci=show_ci)
        st.plotly_chart(fig1, use_container_width=True, config={'displayModeBar': False}, theme=None)

if show_ml:
    with col2:
        fig2 = _build_chart_plotly("Метод ML (Isolation Forest)", anomalies_ml, "Найдено", method_type="ml", details=details_ml, show_ci=show_ci)
        st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False}, theme=None)


# Метрики
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

