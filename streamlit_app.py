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
        try:
            return pd.read_csv(file)
        except Exception as e:
            st.error(f"Erro ao processar o arquivo .csv: {e}")
            return None
    elif file.name.endswith('.fit'):
        data = []
        try:
            with fitdecode.FitReader(file) as fitfile:
                for frame in fitfile:
                    if isinstance(frame, fitdecode.records.FitDataMessage):  # Processar apenas mensagens de dados
                        record = {field.name: field.value for field in frame.fields}
                        data.append(record)
            if data:
                return pd.DataFrame(data)
            else:
                st.error("Nenhum dado encontrado no arquivo .fit.")
                return None
        except Exception as e:
            st.error(f"Erro ao processar o arquivo .fit: {e}")
            return None
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
        st.write("### Colunas Pré-Selecionadas:")
        st.write(column_map)

        # Seleção da coluna de tempo
        st.write("### Selecione o Intervalo de Tempo para Análise")

        # Procurar colunas que começam com "time" ou "timestamp"
        time_columns = [col for col in df.columns if "time" in col.lower() or "timestamp" in col.lower()]

        if not time_columns:
            st.error("Nenhuma coluna com o nome 'time' ou 'timestamp' foi encontrada.")
            st.stop()
        else:
            # Usar automaticamente a primeira coluna encontrada
            time_column_name = time_columns[0]
            time_column = df[time_column_name]
            st.write(f"Coluna de tempo detectada: **{time_column_name}**")
            
            # Verificar se a coluna contém data e hora combinadas
            if pd.api.types.is_string_dtype(time_column):
                try:
                    # Converter strings de data e hora para objetos datetime
                    time_column = pd.to_datetime(time_column)
                    # Extrair apenas o tempo em segundos
                    time_column = time_column.dt.hour * 3600 + time_column.dt.minute * 60 + time_column.dt.second
                    st.success(f"Coluna de tempo processada e convertida para segundos.")
                except Exception as e:
                    st.error(f"Erro ao processar a coluna de tempo: {e}")
                    st.stop()
            elif not pd.api.types.is_numeric_dtype(time_column):
                st.error("A coluna selecionada para o tempo não é numérica nem uma string válida de data e hora.")
                st.stop()

        # Criar o slider de tempo com valores mínimos e máximos válidos
        min_time, max_time = st.slider(
            "Intervalo de Tempo",
            min_value=int(time_column.min()),
            max_value=int(time_column.max()),
            value=(int(time_column.min()), int(time_column.max()))
        )

        # Filtrar os dados com base no intervalo selecionado
        df_filtered = df[(time_column >= min_time) & (time_column <= max_time)]

        # Mostrar gráfico com colunas selecionadas antes do filtro
        st.write("### Gráfico Antes do Filtro")
        available_columns = df.columns.tolist()
        selected_columns = st.multiselect(
            "Escolha as colunas para incluir no gráfico:",
            available_columns,
            default=list(column_map.values())
        )
        fig = go.Figure()
        for col in selected_columns:
            fig.add_trace(go.Scatter(x=time_column, y=df_filtered[col], mode='lines', name=col))
        
        fig.update_layout(
            xaxis=dict(title="Tempo"),
            title="Gráfico Antes do Filtro",
            legend=dict(orientation="h")
        )
        st.plotly_chart(fig)

        # Aplicar Filtros
        st.write("### Aplicar Filtros")
        default_cutoff = {"SmO2": 0.1, "THb": 0.2, "Power": 0.1, "HR": 0.1}
        filtered_columns = {}
        for col in selected_columns:
            cutoff = default_cutoff.get(col, 0.1)
            if col in df.columns:
                filtered_col_name = f"{col}_filtered"
                df_filtered[filtered_col_name] = butterworth_filter(df_filtered[col].interpolate(), cutoff=cutoff)
                filtered_columns[col] = filtered_col_name
                st.write(f"Filtro Butterworth aplicado em **{col}** (Frequência de corte: {cutoff} Hz).")

        # Gráficos Pós-Filtro
        st.write("### Visualizar Gráficos Pós-Filtro")
        selected_col = st.selectbox("Selecione uma coluna para visualizar:", filtered_columns.keys())
        if selected_col:
            filtered_col_name = filtered_columns[selected_col]
            fig_filtered = go.Figure()
            fig_filtered.add_trace(go.Scatter(x=time_column, y=df_filtered[selected_col], mode='lines', name="Original"))
            fig_filtered.add_trace(go.Scatter(x=time_column, y=df_filtered[filtered_col_name], mode='lines', name="Filtrado"))
            fig_filtered.update_layout(title=f"Gráfico de {selected_col} Pós-Filtro")
            st.plotly_chart(fig_filtered)

        # Download dos dados processados
        st.write("### Baixar Dados Processados")
        csv = df_filtered.to_csv(index=False).encode('utf-8')
        st.download_button("Baixar CSV Filtrado", data=csv, file_name="dados_filtrados.csv", mime="text/csv")
