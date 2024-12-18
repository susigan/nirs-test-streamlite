import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import fitdecode
import plotly.graph_objects as go

# Função para aplicar o filtro Butterworth
def butterworth_filter(data, cutoff=0.1, fs=1.0, order=2):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# Função para carregar arquivo .fit ou .csv
def load_file(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith('.fit'):
        data = []
        with fitdecode.FitReader(file) as fitfile:
            for frame in fitfile:
                if isinstance(frame, fitdecode.records.FitDataMessage):
                    data.append({field.name: field.value for field in frame.fields})
        return pd.DataFrame(data)
    else:
        st.error("Formato de arquivo não suportado. Use '.csv' ou '.fit'.")
        return None

# Função para detectar colunas importantes
def detect_columns(df):
    known_columns = {
        "SmO2": ["saturated_hemoglobin_percent", "SmO2"],
        "THb": ["total_hemoglobin_con", "THb"],
        "Power": ["power"],
        "HR": ["heart_rate", "hr"]
    }
    column_map = {}
    for key, possible_names in known_columns.items():
        for name in possible_names:
            matched_columns = [col for col in df.columns if name.lower() in col.lower()]
            if matched_columns:
                column_map[key] = matched_columns[0]
                break
    return column_map

# Interface do Streamlit
st.title("Análise NIRS e Dados Fisiológicos")
st.write("Faça upload de arquivos `.csv` ou `.fit` para analisar os dados.")

# Upload do arquivo
uploaded_file = st.file_uploader("Carregue seu arquivo (.csv ou .fit):", type=["csv", "fit"])

if uploaded_file:
    df = load_file(uploaded_file)
    if df is not None:
        st.write("Prévia dos dados:")
        st.dataframe(df.head())

        # Detectar colunas automaticamente
        column_map = detect_columns(df)

        # Procurar colunas de tempo
        time_columns = [col for col in df.columns if "time" in col.lower() or "timestamp" in col.lower()]
        time_column = pd.to_datetime(df[time_columns[0]], errors="coerce").dt.total_seconds()

        min_time, max_time = st.slider("Intervalo de Tempo", int(time_column.min()), int(time_column.max()), (int(time_column.min()), int(time_column.max())))
        df_filtered = df[(time_column >= min_time) & (time_column <= max_time)].copy()

        # Aplicar filtros
        default_cutoff = {"SmO2": 0.1, "Power": 0.1, "HR": 0.1}
        filtered_columns = {}
        for col_key, cutoff in default_cutoff.items():
            if col_key in column_map:
                col_name = column_map[col_key]
                df_filtered[f"{col_name}_filtered"] = butterworth_filter(df_filtered[col_name].interpolate(), cutoff=cutoff)
                filtered_columns[col_key] = f"{col_name}_filtered"

        # Steps
        work_time = st.text_input("Tempo de Trabalho (mm:ss):", "02:00")
        rest_time = st.text_input("Tempo de Descanso (mm:ss):", "01:00")

        def time_to_seconds(time_str):
            try:
                minutes, seconds = map(int, time_str.split(":"))
                return minutes * 60 + seconds
            except ValueError:
                st.error("Formato inválido! Use mm:ss")
                return None

        work_seconds = time_to_seconds(work_time)
        rest_seconds = time_to_seconds(rest_time)

        if work_seconds and rest_seconds:
            steps = []
            current_time = min_time
            while current_time < max_time:
                steps.append({"Type": "Trabalho", "Start": current_time, "End": min(current_time + work_seconds, max_time)})
                current_time += work_seconds
                if current_time < max_time:
                    steps.append({"Type": "Descanso", "Start": current_time, "End": min(current_time + rest_seconds, max_time)})
                    current_time += rest_seconds

            # Gráfico
            st.write("### Gráfico com Steps")
            fig = go.Figure()

            # Adicionar dados filtrados primeiro
            for col_key, col_name in filtered_columns.items():
                fig.add_trace(go.Scatter(
                    x=time_column, y=df_filtered[col_name], mode='lines', name=col_key
                ))

            # Adicionar Steps no fundo
            for step in steps:
                fig.add_vrect(
                    x0=step["Start"], x1=step["End"],
                    fillcolor="green" if step["Type"] == "Trabalho" else "red",
                    opacity=0.2, layer="below", line_width=0
                )

            fig.update_layout(
                xaxis=dict(title="Tempo (segundos)", range=[min_time, max_time]),
                yaxis=dict(title="Valores"),
                title="Gráfico com Steps de Trabalho e Descanso",
                legend=dict(orientation="h")
            )
            st.plotly_chart(fig)
