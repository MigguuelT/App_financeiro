# ==============================================================================
# PROJETO: ANALISADOR QUANTITATIVO MULTIVARIADO (ATIVO + MACRO)
# DESENVOLVEDOR: MIG / COLABORADORA: GEMINI
# DATA: 23/03/2026
# ==============================================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

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
st.title("📈 Inteligência Quantitativa: Análise Multivariada e IA")

# --- BARRA LATERAL ---
st.sidebar.header("Parâmetros Institucionais")
ticker_symbol = st.sidebar.text_input("Ativo Principal (ex: IAU, VALE3.SA):", "IAU").upper()
anos_historico = st.sidebar.number_input("Histórico (Anos):", 1, 20, 10)
dias_predicao = st.sidebar.slider("Janela de Predição (Dias):", 10, 90, 30)

st.sidebar.markdown("---")
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
else:
    api_key = st.sidebar.text_input("Chave API (Gemini):", type="password")
st.sidebar.markdown("---")

# ==============================================================================
# FUNÇÃO 1: COLETA MULTIVARIADA (SINCRONIZAÇÃO DE ATIVOS)
# ==============================================================================

@st.cache_data(ttl="1d", show_spinner=False)
def carregar_dados_completos(ticker_principal, anos):
    """Sincroniza o ativo principal com um indexador macro (Dólar)."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=anos * 365)
    
    # Define o par correlacionado (Dólar Global ou Câmbio BRL)
    ticker_macro = "UUP" if not ticker_principal.endswith(".SA") else "USDBRL=X"
    
    # Download conjunto para garantir as mesmas datas
    try:
        data = yf.download([ticker_principal, ticker_macro], 
                           start=start_date.strftime('%Y-%m-%d'), 
                           end=end_date.strftime('%Y-%m-%d'), 
                           progress=False)['Close']
        
        data = data.reset_index().dropna().ffill()
        # Renomeação dinâmica para garantir consistência no ML
        data.columns = ['Date', 'Macro', 'Ativo']
        
        # Criação de retornos estacionários
        data['Ret_Ativo'] = data['Ativo'].pct_change()
        data['Ret_Macro'] = data['Macro'].pct_change()
        data['Correl_30d'] = data['Ret_Ativo'].rolling(30).corr(data['Ret_Macro'])
        data['Year'] = data['Date'].dt.year
        
        return data.dropna(), ticker_macro
    except:
        return pd.DataFrame(), ""

# ==============================================================================
# FUNÇÃO 2: XGBOOST MULTIVARIADO (RECURSOS CRUZADOS)
# ==============================================================================

@st.cache_resource(show_spinner=False)
def treinar_xgboost_multi(df, dias_futuros):
    df_ml = df.copy()
    
    # Features do Ativo + Features do Macro (Visão Periférica)
    for lag in [1, 5, 21]:
        df_ml[f'Lag_Ativo_{lag}'] = df_ml['Ret_Ativo'].shift(lag)
        df_ml[f'Lag_Macro_{lag}'] = df_ml['Ret_Macro'].shift(lag)
        
    # Distâncias de Médias Móveis (Estacionárias)
    df_ml['Dist_SMA_200'] = (df_ml['Ativo'] - df_ml['Ativo'].rolling(200).mean()) / df_ml['Ativo'].rolling(200).mean()
    
    # Target Multiclasse (0: Baixa, 1: Neutro, 2: Alta)
    ret_futuro = (df_ml['Ativo'].shift(-dias_futuros) - df_ml['Ativo']) / df_ml['Ativo']
    limite = df_ml['Ret_Ativo'].std() * np.sqrt(dias_futuros) * 0.5
    df_ml['Target'] = np.select([(ret_futuro > limite), (ret_futuro < -limite)], [2, 0], default=1)
    
    df_ml = df_ml.dropna()
    features = [c for c in df_ml.columns if 'Lag' in c or 'Correl' in c or 'Dist' in c]
    
    df_t = df_ml.iloc[:-dias_futuros]
    split = int(len(df_t) * 0.8)
    X_train, X_test = df_t[features][:split], df_t[features][split:]
    y_train, y_test = df_t['Target'][:split], df_t['Target'][split:]
    
    model = xgb.XGBClassifier(objective='multi:softprob', num_class=3, n_estimators=150, learning_rate=0.05)
    model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, model.predict(X_test))
    prob = model.predict_proba(df_ml.iloc[-1:][features])[0] * 100
    
    return prob, acc

# ==============================================================================
# FUNÇÃO 3: LSTM MULTIVARIADA (TENSORES DE TEMPO E ESPAÇO)
# ==============================================================================

@st.cache_resource(show_spinner=False)
def treinar_lstm_multi(df, dias_futuros):
    scaler = StandardScaler()
    # A rede agora "vê" duas colunas por vez
    scaled_f = scaler.fit_transform(df[['Ret_Ativo', 'Ret_Macro']])
    
    ret_futuro = (df['Ativo'].shift(-dias_futuros) - df['Ativo']) / df['Ativo']
    limite = df['Ret_Ativo'].std() * np.sqrt(dias_futuros) * 0.5
    targets = np.select([(ret_futuro > limite), (ret_futuro < -limite)], [2, 0], default=1)
    
    lookback = 60
    X, y = [], []
    for i in range(lookback, len(scaled_f) - dias_futuros):
        X.append(scaled_f[i-lookback:i]) # Janela de 60 dias com 2 variáveis
        y.append(targets[i])
    
    X, y = np.array(X), np.array(y)
    split = int(len(X) * 0.8)
    
    model = Sequential([
        LSTM(50, input_shape=(lookback, 2)), # 2 canais de entrada
        Dropout(0.2),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X[:split], y[:split], validation_data=(X[split:], y[split:]), epochs=30, batch_size=32, verbose=0)
    
    acc = accuracy_score(y[split:], np.argmax(model.predict(X[split:], verbose=0), axis=1))
    prob = model.predict(np.reshape(scaled_f[-lookback:], (1, lookback, 2)), verbose=0)[0] * 100
    
    return prob, acc

# ==============================================================================
# FRONT-END E EXIBIÇÃO
# ==============================================================================

if st.sidebar.button("Analisar Ativo"):
    df_full, ticker_m = carregar_dados_completos(ticker_symbol, anos_historico)
    
    if not df_full.empty:
        try: moeda = yf.Ticker(ticker_symbol).fast_info.currency
        except: moeda = "USD"

        aba_eda, aba_ml, aba_ia = st.tabs(["📊 Exploração & Correlação", "🤖 Modelos Multivariados", "🧠 Agente Financeiro"])

        with aba_eda:
            st.subheader(f"Análise Cruzada: {ticker_symbol} vs {ticker_m}")
            c1, c2 = st.columns(2)
            c1.metric("Preço Atual", f"{moeda} {df_full['Ativo'].iloc[-1]:.2f}")
            # Correlação Atual (Últimos 30 dias)
            correl_atual = df_full['Correl_30d'].iloc[-1]
            c2.metric("Correlação (30d)", f"{correl_atual:.2f}", delta="Inversa" if correl_atual < 0 else "Direta")

            st.markdown("---")
            st.subheader("Gráfico Comparativo Normalizado")
            # Normalização (Base 100) para comparar ativos de preços diferentes
            df_norm = df_full.tail(500).copy()
            df_norm['Ativo_N'] = (df_norm['Ativo'] / df_norm['Ativo'].iloc[0]) * 100
            df_norm['Macro_N'] = (df_norm['Macro'] / df_norm['Macro'].iloc[0]) * 100
            
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Scatter(x=df_norm['Date'], y=df_norm['Ativo_N'], name=ticker_symbol, line=dict(color='blue')))
            fig_comp.add_trace(go.Scatter(x=df_norm['Date'], y=df_norm['Macro_N'], name=f"{ticker_m} (Dólar)", line=dict(color='red', dash='dot')))
            fig_comp.update_layout(template="plotly_white", height=400, yaxis_title="Base 100")
            st.plotly_chart(fig_comp, use_container_width=True)

            # Tabela Estilizada (Restaurada)
            st.markdown("---")
            df_y = df_full.groupby('Year')['Ativo'].mean().reset_index()
            df_y['Var %'] = df_y['Ativo'].pct_change() * 100
            st.dataframe(df_y.style.background_gradient(cmap='RdYlGn', subset=['Var %']), use_container_width=True)

        with aba_ml:
            with st.spinner("Treinando modelos multivariados..."):
                p_xgb, a_xgb = treinar_xgboost_multi(df_full, dias_predicao)
                p_lstm, a_lstm = treinar_lstm_multi(df_full, dias_predicao)

            st.subheader(f"Probabilidade Direcional ({dias_predicao} dias)")
            def bar_p(probs, title):
                fig = go.Figure(go.Bar(x=[probs[0]], name='Baixa', orientation='h', marker_color='#ffcccb'))
                fig.add_trace(go.Bar(x=[probs[1]], name='Neutro', orientation='h', marker_color='#f0f0f0'))
                fig.add_trace(go.Bar(x=[probs[2]], name='Alta', orientation='h', marker_color='#d4edda'))
                fig.update_layout(barmode='stack', height=180, title=title, xaxis=dict(range=[0,100], ticksuffix='%'))
                return fig

            st.plotly_chart(bar_p(p_xgb, f"XGBoost Multivariado (Acc: {a_xgb*100:.1f}%)"), use_container_width=True)
            st.plotly_chart(bar_p(p_lstm, f"LSTM Multivariada (Acc: {a_lstm*100:.1f}%)"), use_container_width=True)

        with aba_ia:
            if api_key:
                genai.configure(api_key=api_key)
                model_ia = genai.GenerativeModel('gemini-2.5-pro')
                ctx = f"Ativo: {ticker_symbol}. Macro: {ticker_m}. Correl: {correl_atual:.2f}. XGB: {p_xgb[2]:.1f}% Alta. LSTM: {p_lstm[2]:.1f}% Alta."
                st.write(model_ia.generate_content(f"Relatório Executivo Quant: {ctx}").text)