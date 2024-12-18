import streamlit as st
import pandas as pd
import json
import numpy as np
from scipy.signal import butter, filtfilt
import fitdecode
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Função para aplicar o filtro Butterworth
def butterworth_filter(data, cutoff=0.1, fs=1.0, order=2):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# Função para carregar arquivos
def load_file(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith('.fit'):
        data = []
        with fitdecode.FitReader(file) as fitfile:
            for frame in fitfile:
                if frame.frame_type == fitdecode.FIT_FRAME_DATA:
                    data.append(frame.get_values())
        return pd.DataFrame(data)
    else:
        st.error("Formato de arquivo não suportado. Use '.csv' ou '.fit'.")
        return None

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

        # Selecionar colunas para filtragem
        st.write("### Aplicar Filtro Butterworth")
        columns = st.multiselect("Selecione as colunas para filtrar:", df.columns)
        cutoff = st.slider("Defina a frequência de corte:", min_value=0.01, max_value=1.0, value=0.1)

        for col in columns:
            if df[col].dtype in ['float64', 'int64']:
                df[f"{col}_filtered"] = butterworth_filter(df[col].interpolate(), cutoff=cutoff)

        st.write("Dados filtrados:")
        st.dataframe(df.head())

        # Plotar os dados filtrados
        st.write("### Gráficos")
        selected_col = st.selectbox("Selecione uma coluna para visualizar:", columns)
        if selected_col:
            st.line_chart(df[[selected_col, f"{selected_col}_filtered"]])

        # Download dos dados processados
        st.write("### Baixar Dados Filtrados")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Baixar CSV Filtrado", data=csv, file_name="dados_filtrados.csv", mime="text/csv")
