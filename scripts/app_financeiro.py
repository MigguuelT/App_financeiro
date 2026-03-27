import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import os
import json
import plotly.graph_objects as go
import google.generativeai as genai

# --- CONFIGURAÇÃO DE DIRETÓRIOS ---
DIRETORIO_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
RAIZ_PROJETO = os.path.abspath(os.path.join(DIRETORIO_SCRIPTS, ".."))
PASTA_DATA = os.path.join(RAIZ_PROJETO, "data")
PASTA_MODELS = os.path.join(RAIZ_PROJETO, "models")

st.set_page_config(page_title="Quant Terminal Pro", layout="wide", page_icon="📈")

# --- CARREGAMENTO DE ASSETS E MÉTRICAS ---
@st.cache_resource
def carregar_tudo(ticker="qqq"):
    try:
        path_csv = os.path.join(PASTA_DATA, f"{ticker}_processed.csv")
        path_metrics = os.path.join(PASTA_MODELS, f"metricas_{ticker}.json")
        
        if not os.path.exists(path_csv) or not os.path.exists(path_metrics):
            return None

        with open(path_metrics, "r") as f:
            m = json.load(f)

        return {
            "df": pd.read_csv(path_csv, index_col=0, parse_dates=True),
            "xgb": joblib.load(os.path.join(PASTA_MODELS, f"modelo_xgb_{ticker}.joblib")),
            "lstm": tf.keras.models.load_model(os.path.join(PASTA_MODELS, f"modelo_lstm_{ticker}.keras")),
            "scaler": joblib.load(os.path.join(PASTA_MODELS, f"scaler_{ticker}.joblib")),
            "metricas": m
        }
    except Exception as e:
        st.error(f"Erro ao carregar: {e}")
        return None

# --- INTERFACE PRINCIPAL ---
st.title("📈 Terminal Quantitativo & Consultoria IA")
ticker_alvo = "qqq"
assets = carregar_tudo(ticker_alvo)

if assets:
    df = assets["df"]
    m = assets["metricas"]
    
    aba_eda, aba_ml, aba_ia = st.tabs(["📊 Exploração", "🤖 Machine Learning", "🧠 Consultoria IA"])

    with aba_eda:
        st.subheader(f"Análise Exploratória: {ticker_alvo.upper()}")
        c1, c2, c3, c4 = st.columns(4)
        preco_atual = df['Ativo'].iloc[-1]
        c1.metric("Preço Atual", f"$ {preco_atual:.2f}")
        c2.metric("Média Histórica", f"$ {df['Ativo'].mean():.2f}")
        c3.metric("Volatilidade (21d)", f"{df['Volatilidade'].iloc[-1]*100:.2f}%")
        c4.metric("Data do Treino", m['data_treino'])

        fig_precos = go.Figure()
        fig_precos.add_trace(go.Scatter(x=df.index, y=df['Ativo'], name="Fechamento", line=dict(color='#1f77b4')))
        fig_precos.update_layout(template="plotly_white", height=450, title="Histórico de Preços (Base de Treino)", xaxis_title="Data", yaxis_title="Preço (USD)")
        st.plotly_chart(fig_precos, use_container_width=True)

    with aba_ml:
        st.subheader("Predições Baseadas em Modelos Treinados")
        
        # Inferencia XGBoost
        f_xgb = [c for c in df.columns if 'Lag' in c or 'Volatilidade' in c]
        dados_xgb = df[f_xgb].tail(1)
        p_xgb = assets["xgb"].predict_proba(dados_xgb)[0] * 100
        
        # Inferencia LSTM
        f_lstm = ['Ret_Ativo', 'Ret_Macro', 'Volatilidade']
        dados_norm = assets["scaler"].transform(df[f_lstm].tail(60))
        p_lstm = assets["lstm"].predict(np.reshape(dados_norm, (1, 60, 3)), verbose=0)[0] * 100

        def plot_prob(probs, titulo, acc):
            fig = go.Figure()
            fig.add_trace(go.Bar(y=[''], x=[probs[0]], name='Baixa', orientation='h', marker_color='#ffcccb', width=0.6))
            fig.add_trace(go.Bar(y=[''], x=[probs[1]], name='Neutro', orientation='h', marker_color='#f0f0f0', width=0.6))
            fig.add_trace(go.Bar(y=[''], x=[probs[2]], name='Alta', orientation='h', marker_color='#d4edda', width=0.6))
            fig.update_layout(barmode='stack', height=180, title=f"{titulo} (Acurácia de Treino: {acc*100:.1f}%)", 
                              xaxis=dict(range=[0,100], ticksuffix='%'), margin=dict(l=0, r=0, t=40, b=20))
            return fig

        st.plotly_chart(plot_prob(p_xgb, "XGBoost (Classificador Tabular)", m['acuracia_xgb']), use_container_width=True)
        st.plotly_chart(plot_prob(p_lstm, "LSTM (Rede Neural Sequencial)", m['acuracia_lstm']), use_container_width=True)

    with aba_ia:
        st.subheader("Relatório Estratégico do Gestor IA")
        api_key = st.text_input("Insira sua Gemini API Key:", type="password")
        
        if st.button("Gerar Parecer Técnico"):
            if api_key:
                try:
                    genai.configure(api_key=api_key)
                    model_ia = genai.GenerativeModel('gemini-2.0-flash')
                    prompt = f"""
                    Atue como Gestor Sênior de Investimentos. Analise os seguintes dados quantitativos para {ticker_alvo.upper()}:
                    - Preço Atual: ${preco_atual:.2f}
                    - Modelo XGBoost: {p_xgb[2]:.1f}% de probabilidade de ALTA (Acurácia: {m['acuracia_xgb']*100:.1f}%)
                    - Modelo LSTM: {p_lstm[2]:.1f}% de probabilidade de ALTA (Acurácia: {m['acuracia_lstm']*100:.1f}%)
                    - Volatilidade Atual: {df['Volatilidade'].iloc[-1]*100:.2f}%
                    
                    Gere um parecer direto, sem saudações, focado em alocação (COMPRA, MANUTENÇÃO ou REDUÇÃO).
                    """
                    resposta = model_ia.generate_content(prompt)
                    st.markdown("---")
                    st.markdown(resposta.text)
                except Exception as e:
                    st.error(f"Erro na IA: {e}")
            else:
                st.warning("⚠️ Chave API necessária para o parecer.")

else:
    st.info("🔄 Aguardando primeira execução da DAG no Airflow para carregar modelos e métricas.")