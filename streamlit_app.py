import streamlit as st

# Título do aplicativo
st.title("Meu Primeiro App no Streamlit")

# Texto introdutório
st.write("Este é meu primeiro aplicativo interativo no Streamlit Community Cloud!")

# Caixa de entrada para o usuário
nome = st.text_input("Qual é o seu nome?")
if nome:
    st.write(f"Olá, {nome}! Seja bem-vindo(a) ao meu aplicativo!")
