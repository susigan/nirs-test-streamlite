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
    st.success("Arquivo carregado com sucesso!")
    df = load_file(uploaded_file)

    if df is not None:
        st.write("Prévia dos dados:")
        st.dataframe(df.head())

        # Detectar colunas automaticamente
        column_map = detect_columns(df)
        st.write("### Colunas Detectadas:")
        st.write(column_map)

        # Slider para selecionar intervalo de tempo
        st.write("### Selecione o Intervalo de Tempo para Análise")
        time_column = df.index if "time" not in df.columns else df["time"]
        min_time, max_time = st.slider(
            "Intervalo de Tempo",
            min_value=int(time_column.min()),
            max_value=int(time_column.max()),
            value=(int(time_column.min()), int(time_column.max()))
        )

        # Filtrar dados no intervalo selecionado
        df_filtered = df[(time_column >= min_time) & (time_column <= max_time)]

        # Mostrar gráfico com Power, SmO2 e HR antes do filtro
        st.write("### Gráfico Antes do Filtro")
        fig = go.Figure()
        if "Power" in column_map:
            fig.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered[column_map["Power"]], mode='lines', name="Power (Y1)", yaxis="y1"))
        if "SmO2" in column_map:
            fig.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered[column_map["SmO2"]], mode='lines', name="SmO2 (Y2)", yaxis="y2"))
        if "HR" in column_map:
            fig.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered[column_map["HR"]], mode='lines', name="HR (Y3)", yaxis="y3"))
        
        # Configurações dos eixos do gráfico
        fig.update_layout(
            xaxis=dict(title="Tempo"),
            yaxis=dict(title="Power (Y1)", side="left"),
            yaxis2=dict(title="SmO2 (Y2)", overlaying="y", side="right"),
            yaxis3=dict(title="HR (Y3)", anchor="free", position=0.85),
            title="Gráfico Antes do Filtro",
            legend=dict(orientation="h")
        )
        st.plotly_chart(fig)

        # Filtros
        st.write("### Aplicar Filtros")
        default_cutoff = {"SmO2": 0.1, "THb": 0.2, "Power": 0.1, "HR": 0.1}
        filtered_columns = {}
        for col_key, cutoff in default_cutoff.items():
            if col_key in column_map:
                col_name = column_map[col_key]
                df_filtered[f"{col_name}_filtered"] = butterworth_filter(df_filtered[col_name].interpolate(), cutoff=cutoff)
                filtered_columns[col_key] = f"{col_name}_filtered"
                st.write(f"Filtro Butterworth aplicado em **{col_name}** (Frequência de corte: {cutoff} Hz).")

        # Gráficos interativos pós-filtro
        st.write("### Visualizar Gráficos Pós-Filtro")
        selected_col = st.selectbox("Selecione uma coluna para visualizar:", filtered_columns.keys())
        if selected_col:
            filtered_col_name = filtered_columns[selected_col]
            fig_filtered = go.Figure()
            fig_filtered.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered[column_map[selected_col]], mode='lines', name="Original"))
            fig_filtered.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered[filtered_col_name], mode='lines', name="Filtrado"))
            fig_filtered.update_layout(title=f"Gráfico de {selected_col} Pós-Filtro")
            st.plotly_chart(fig_filtered)

        # Download dos dados processados
        st.write("### Baixar Dados Processados")
        csv = df_filtered.to_csv(index=False).encode('utf-8')
        st.download_button("Baixar CSV Filtrado", data=csv, file_name="dados_filtrados.csv", mime="text/csv")
