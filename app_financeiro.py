# ==============================================================================
# PROJETO: ANALISADOR QUANTITATIVO MULTIVARIADO (ESTÁVEL & ESTRATÉGICO)
# DESENVOLVEDOR: MIG / COLABORADORA: GEMINI
# DATA: 23/03/2026
# ==============================================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# Modelagem e Métricas
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Deep Learning (TensorFlow)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# IA Generativa
import google.generativeai as genai

# --- CONFIGURAÇÃO DA INTERFACE ---
st.set_page_config(page_title="Quant Multivariado & IA", layout="wide")
st.title("📈 Inteligência Quantitativa: Estratégia Multivariada e IA")

# --- BARRA LATERAL ---
st.sidebar.header("Parâmetros Institucionais")
ticker_symbol = st.sidebar.text_input("Ativo Principal:", "IAU").upper()
anos_historico = st.sidebar.number_input("Histórico (Anos):", 1, 20, 10)
dias_predicao = st.sidebar.slider("Janela de Predição (Dias):", 10, 90, 30)

st.sidebar.markdown("---")
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
else:
    api_key = st.sidebar.text_input("Chave API (Gemini):", type="password")
st.sidebar.markdown("---")

# ==============================================================================
# FUNÇÃO 1: COLETA MULTIVARIADA (TRATAMENTO DE ERROS E RATE LIMIT)
# ==============================================================================

@st.cache_data(ttl="1h", show_spinner=False)
def carregar_dados_completos(ticker_principal, anos):
    """Sincroniza o ativo principal com um indexador macro (Dólar)."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=anos * 365)
    ticker_macro = "UUP" if not ticker_principal.endswith(".SA") else "USDBRL=X"
    
    try:
        # Download conjunto
        data = yf.download([ticker_principal, ticker_macro], 
                           start=start_date.strftime('%Y-%m-%d'), 
                           end=end_date.strftime('%Y-%m-%d'), 
                           progress=False)['Close']
        
        # Fallback caso o macro falhe (Rate Limit)
        if ticker_macro not in data.columns:
            st.warning(f"⚠️ Limite de API para {ticker_macro}. Usando modo Univariado.")
            data = data[[ticker_principal]].copy()
            data.columns = ['Ativo']
            data['Macro'] = data['Ativo'] 
        else:
            data = data.reset_index().dropna().ffill()
            data.columns = ['Date', 'Macro', 'Ativo']
            
        data['Ret_Ativo'] = data['Ativo'].pct_change()
        data['Ret_Macro'] = data['Macro'].pct_change()
        data['Correl_30d'] = data['Ret_Ativo'].rolling(30).corr(data['Ret_Macro'])
        data['Year'] = data['Date'].dt.year
        
        return data.dropna(), ticker_macro
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return pd.DataFrame(), ""

# ==============================================================================
# FUNÇÕES DE MACHINE LEARNING (ESTACIONÁRIAS E MULTICLASSE)
# ==============================================================================

@st.cache_resource(show_spinner=False)
def treinar_xgboost_multi(df, dias_futuros):
    df_ml = df.copy()
    for lag in [1, 5, 21]:
        df_ml[f'Lag_Ativo_{lag}'] = df_ml['Ret_Ativo'].shift(lag)
        df_ml[f'Lag_Macro_{lag}'] = df_ml['Ret_Macro'].shift(lag)
    
    df_ml['Dist_SMA_200'] = (df_ml['Ativo'] - df_ml['Ativo'].rolling(200).mean()) / df_ml['Ativo'].rolling(200).mean()
    
    ret_futuro = (df_ml['Ativo'].shift(-dias_futuros) - df_ml['Ativo']) / df_ml['Ativo']
    limite = df_ml['Ret_Ativo'].std() * np.sqrt(dias_futuros) * 0.5
    df_ml['Target'] = np.select([(ret_futuro > limite), (ret_futuro < -limite)], [2, 0], default=1)
    
    df_ml = df_ml.dropna()
    features = [c for c in df_ml.columns if 'Lag' in c or 'Correl' in c or 'Dist' in c]
    
    df_t = df_ml.iloc[:-dias_futuros]
    split = int(len(df_t) * 0.8)
    model = xgb.XGBClassifier(objective='multi:softprob', num_class=3, n_estimators=100)
    model.fit(df_t[features][:split], df_t['Target'][:split])
    
    acc = accuracy_score(df_t['Target'][split:], model.predict(df_t[features][split:]))
    prob = model.predict_proba(df_ml.iloc[-1:][features])[0] * 100
    return prob, acc

@st.cache_resource(show_spinner=False)
def treinar_lstm_multi(df, dias_futuros):
    scaler = StandardScaler()
    scaled_f = scaler.fit_transform(df[['Ret_Ativo', 'Ret_Macro']])
    
    ret_f = (df['Ativo'].shift(-dias_futuros) - df['Ativo']) / df['Ativo']
    limite = df['Ret_Ativo'].std() * np.sqrt(dias_futuros) * 0.5
    targets = np.select([(ret_f > limite), (ret_f < -limite)], [2, 0], default=1)
    
    lookback = 60
    X, y = [], []
    for i in range(lookback, len(scaled_f) - dias_futuros):
        X.append(scaled_f[i-lookback:i])
        y.append(targets[i])
    
    X, y = np.array(X), np.array(y)
    split = int(len(X) * 0.8)
    
    model = Sequential([
        LSTM(32, input_shape=(lookback, 2)),
        Dropout(0.2),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X[:split], y[:split], epochs=20, batch_size=32, verbose=0)
    
    acc = accuracy_score(y[split:], np.argmax(model.predict(X[split:], verbose=0), axis=1))
    prob = model.predict(np.reshape(scaled_f[-lookback:], (1, lookback, 2)), verbose=0)[0] * 100
    return prob, acc

# ==============================================================================
# FRONT-END (INTERFACE E RELATÓRIO ESTRATÉGICO)
# ==============================================================================

if st.sidebar.button("Analisar Ativo"):
    df_full, ticker_m = carregar_dados_completos(ticker_symbol, anos_historico)
    
    if not df_full.empty:
        try: moeda = yf.Ticker(ticker_symbol).fast_info.currency
        except: moeda = "USD"

        aba_eda, aba_ml, aba_ia = st.tabs(["📊 Exploração", "🤖 Machine Learning", "🧠 Consultoria IA"])

        with aba_eda:
            st.subheader(f"Métricas: {ticker_symbol}")
            c1, c2 = st.columns(2)
            c1.metric("Preço Atual", f"{moeda} {df_full['Ativo'].iloc[-1]:.2f}")
            correl = df_full['Correl_30d'].iloc[-1]
            c2.metric("Correlação vs Macro", f"{correl:.2f}")

            st.markdown("---")
            st.subheader("Comparação Anual e Variação")
            df_y = df_full.groupby('Year')['Ativo'].mean().reset_index()
            df_y['Var %'] = df_y['Ativo'].pct_change() * 100
            
            # Ajuste da linha 190: Estilização manual (Sem Matplotlib) 
            def style_negative(v):
                if pd.isna(v): return ""
                color = 'red' if v < 0 else 'green'
                return f'color: {color}; font-weight: bold;'

            st.dataframe(
                df_y.style.applymap(style_negative, subset=['Var %']).format({'Ativo': '{:.2f}', 'Var %': '{:.2f}%'}), 
                width='stretch', 
                hide_index=True
            )

        with aba_ml:
            with st.spinner("Treinando modelos multivariados..."):
                p_xgb, a_xgb = treinar_xgboost_multi(df_full, dias_predicao)
                p_lstm, a_lstm = treinar_lstm_multi(df_full, dias_predicao)

            st.subheader(f"Probabilidades Direcionais ({dias_predicao} dias)")
            def bar_p(probs, title):
                fig = go.Figure(go.Bar(x=[probs[0]], name='Baixa', orientation='h', marker_color='#ffcccb'))
                fig.add_trace(go.Bar(x=[probs[1]], name='Neutro', orientation='h', marker_color='#f0f0f0'))
                fig.add_trace(go.Bar(x=[probs[2]], name='Alta', orientation='h', marker_color='#d4edda'))
                fig.update_layout(barmode='stack', height=180, title=title, xaxis=dict(range=[0,100], ticksuffix='%'), margin=dict(l=0,r=0,t=40,b=20))
                return fig

            st.plotly_chart(bar_p(p_xgb, f"XGBoost Multi (Acc: {a_xgb*100:.1f}%)"), width='stretch')
            st.plotly_chart(bar_p(p_lstm, f"LSTM Multi (Acc: {a_lstm*100:.1f}%)"), width='stretch')

        with aba_ia:
            if api_key:
                try:
                    genai.configure(api_key=api_key)
                    model_ia = genai.GenerativeModel('gemini-2.5-pro')
                    # Prompt enriquecido para sugestão de alocação
                    prompt = f"""
                    Atue como um Gestor de Portfólio Sênior.
                    CONTEXTO: {ticker_symbol} ({moeda}). Indexador: {ticker_m}. Correl: {correl:.2f}.
                    PREDIÇÕES (30 dias):
                    - XGBoost: Alta {p_xgb[2]:.1f}%, Neutro {p_xgb[1]:.1f}%, Baixa {p_xgb[0]:.1f}%.
                    - LSTM: Alta {p_lstm[2]:.1f}%, Neutro {p_lstm[1]:.1f}%, Baixa {p_lstm[0]:.1f}%.
                    TAREFA: Gere um relatório executivo curto.
                    OBRIGATÓRIO: Inclua uma seção final intitulada '### 🎯 Sugestão de Alocação Estratégica' 
                    onde você sugere se o momento é de COMPRA, MANUTENÇÃO ou REDUÇÃO baseada na confluência (ou divergência) dos dados.
                    """
                    st.write(model_ia.generate_content(prompt).text)
                except Exception as e: st.error(f"Erro IA: {e}")