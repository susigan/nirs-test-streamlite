import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import fitdecode
import matplotlib.pyplot as plt
import plotly.express as px

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

# Função para detectar colunas
def detect_columns(df):
    column_map = {}
    for col in df.columns:
        col_lower = col.lower()
        if 'smo2' in col_lower:
            column_map['SmO2'] = col
        elif 'thb' in col_lower:
            column_map['THb'] = col
        elif 'power' in col_lower:
            column_map['Power'] = col
        elif 'heart_rate' in col_lower or 'hr' in col_lower:
            column_map['HR'] = col
    required_columns = {'SmO2', 'THb', 'Power', 'HR'}
    if not required_columns.issubset(column_map.keys()):
        st.warning(f"As colunas necessárias não foram encontradas: {required_columns - set(column_map.keys())}")
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

        # Configurações padrão para filtros
        st.write("### Filtros Automáticos")
        default_cutoff = {"SmO2": 0.1, "THb": 0.2, "Power": 0.1, "HR": 0.1}
        for col_key, cutoff in default_cutoff.items():
            if col_key in column_map:
                col_name = column_map[col_key]
                df[f"{col_name}_filtered"] = butterworth_filter(df[col_name].interpolate(), cutoff=cutoff)
                st.write(f"Filtro Butterworth aplicado em **{col_name}** com frequência de corte {cutoff} Hz.")

        # Gráficos interativos
        st.write("### Visualizar Gráficos")
        selected_col = st.selectbox("Selecione uma coluna para visualizar:", list(column_map.keys()))
        if selected_col:
            col_name = column_map[selected_col]
            fig = px.line(df, x=df.index, y=[col_name, f"{col_name}_filtered"], labels={'value': 'Valores', 'index': 'Tempo'}, title=f"{selected_col} e {selected_col} Filtrado")
            st.plotly_chart(fig)

        # Download dos dados processados
        st.write("### Baixar Dados Processados")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Baixar CSV Filtrado", data=csv, file_name="dados_filtrados.csv", mime="text/csv")
