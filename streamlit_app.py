import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import fitdecode
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
        st.write("### Colunas Selecionadas Automaticamente:")
        st.write(column_map)

        # Mostrar todas as colunas disponíveis
        st.write("### Todas as Colunas Disponíveis:")
        st.write(df.columns.tolist())

        # Permitir ao usuário adicionar colunas extras
        additional_columns = st.multiselect(
            "Selecione mais colunas para filtrar:", 
            df.columns.tolist(), 
            default=list(column_map.values())
        )

        # Configurações padrão para filtros
        st.write("### Filtros Automáticos")
        default_cutoff = {"SmO2": 0.1, "THb": 0.2, "Power": 0.1, "HR": 0.1}
        filtered_columns = {}

        for col_name in additional_columns:
            cutoff = default_cutoff.get(col_name, 0.1)  # Frequência padrão para novas colunas
            if col_name in df.columns:
                filtered_col_name = f"{col_name}_filtered"
                df[filtered_col_name] = butterworth_filter(df[col_name].interpolate(), cutoff=cutoff)
                filtered_columns[col_name] = filtered_col_name
                st.write(f"Filtro Butterworth aplicado em **{col_name}** (Frequência de corte: {cutoff} Hz).")

        # Gráficos interativos
        st.write("### Visualizar Gráficos")
        selected_col = st.selectbox("Selecione uma coluna para visualizar:", filtered_columns.keys())
        if selected_col:
            filtered_col_name = filtered_columns[selected_col]
            fig = px.line(
                df, 
                x=df.index, 
                y=[selected_col, filtered_col_name], 
                labels={'value': 'Valores', 'index': 'Tempo'}, 
                title=f"{selected_col} e {filtered_col_name}"
            )
            st.plotly_chart(fig)

        # Download dos dados processados
        st.write("### Baixar Dados Processados")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Baixar CSV Filtrado", data=csv, file_name="dados_filtrados.csv", mime="text/csv")
