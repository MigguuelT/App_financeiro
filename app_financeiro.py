# ==============================================================================
# PROJETO: ANALISADOR QUANTITATIVO MULTIVARIADO (ESTÁVEL & FLASH IA)
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
# FUNÇÕES DE PROCESSAMENTO E ML
# ==============================================================================

@st.cache_data(ttl="1h", show_spinner=False)
def carregar_dados_completos(ticker_principal, anos):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=anos * 365)
    ticker_macro = "UUP" if not ticker_principal.endswith(".SA") else "USDBRL=X"
    try:
        data = yf.download([ticker_principal, ticker_macro], start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), progress=False)
        
        if 'Close' in data.columns.levels[0]:
            data = data['Close']
            
        if ticker_macro not in data.columns:
            data = data[[ticker_principal]].copy()
            data.columns = ['Ativo']
            data['Macro'] = data['Ativo'] 
        else:
            data = data.reset_index().dropna().ffill()
            # CORREÇÃO CRÍTICA: Renomeando pelas chaves exatas para evitar inversão alfabética
            data = data.rename(columns={ticker_principal: 'Ativo', ticker_macro: 'Macro'})
            # Garantindo a ordem das colunas
            data = data[['Date', 'Macro', 'Ativo']]
            
        df_full_ohlc = yf.download(ticker_principal, start=(end_date - timedelta(days=200)).strftime('%Y-%m-%d'), progress=False)
        if isinstance(df_full_ohlc.columns, pd.MultiIndex):
            df_full_ohlc.columns = df_full_ohlc.columns.get_level_values(0)
            
        data['Ret_Ativo'] = data['Ativo'].pct_change()
        data['Ret_Macro'] = data['Macro'].pct_change()
        data['Correl_30d'] = data['Ret_Ativo'].rolling(30).corr(data['Ret_Macro'])
        data['Year'] = data['Date'].dt.year
        return data.dropna(), ticker_macro, df_full_ohlc
    except Exception as e:
        print(e)
        return pd.DataFrame(), "", pd.DataFrame()

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
    model = Sequential([LSTM(32, input_shape=(lookback, 2)), Dropout(0.2), Dense(3, activation='softmax')])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X[:split], y[:split], epochs=20, batch_size=32, verbose=0)
    acc = accuracy_score(y[split:], np.argmax(model.predict(X[split:], verbose=0), axis=1))
    prob = model.predict(np.reshape(scaled_f[-lookback:], (1, lookback, 2)), verbose=0)[0] * 100
    return prob, acc

# ==============================================================================
# FRONT-END (INTERFACE)
# ==============================================================================

if st.sidebar.button("Analisar Ativo"):
    df_full, ticker_m, df_ohlc = carregar_dados_completos(ticker_symbol, anos_historico)
    
    if not df_full.empty:
        try: moeda = yf.Ticker(ticker_symbol).fast_info.currency
        except: moeda = "USD"

        aba_eda, aba_ml, aba_ia = st.tabs(["📊 Exploração", "🤖 Machine Learning", "🧠 Consultoria IA"])

        with aba_eda:
            st.subheader(f"Métricas Detalhadas: {ticker_symbol}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Preço Atual", f"{moeda} {df_full['Ativo'].iloc[-1]:.2f}")
            c2.metric("Média Histórica", f"{moeda} {df_full['Ativo'].mean():.2f}")
            c3.metric("Máxima", f"{moeda} {df_full['Ativo'].max():.2f}")
            c4.metric("Mínima", f"{moeda} {df_full['Ativo'].min():.2f}")

            st.markdown("---")
            st.subheader("Gráfico de Velas (Últimos 6 Meses)")
            fig_v = go.Figure(data=[go.Candlestick(
                x=df_ohlc.index,
                open=df_ohlc['Open'],
                high=df_ohlc['High'],
                low=df_ohlc['Low'],
                close=df_ohlc['Close'],
                name=ticker_symbol
            )])
            fig_v.update_layout(
                xaxis_rangeslider_visible=False, height=400, template="plotly_white", margin=dict(l=0, r=0, t=30, b=0),
                xaxis_title="Data",
                yaxis_title=f"Preço do Ativo ({moeda})"
            )
            st.plotly_chart(fig_v, use_container_width=True)

            col_d, col_b = st.columns(2)
            with col_d:
                st.subheader("Distribuição de Frequência")
                fig_d = go.Figure(go.Histogram(x=df_full['Ativo'], marker_color='lightblue', opacity=0.7))
                m_avg, m_med = df_full['Ativo'].mean(), df_full['Ativo'].median()
                
                fig_d.add_vline(x=m_avg, line_dash="dash", line_color="red", 
                                annotation_text=f"Média: {m_avg:.2f}", 
                                annotation_position="top right", annotation_yshift=20)
                fig_d.add_vline(x=m_med, line_dash="dash", line_color="green", 
                                annotation_text=f"Mediana: {m_med:.2f}", 
                                annotation_position="top right", annotation_yshift=0)
                
                fig_d.update_layout(template="plotly_white", height=350)
                st.plotly_chart(fig_d, width='stretch')
            with col_b:
                st.subheader("Box Plot de Preços")
                fig_b = go.Figure(go.Box(y=df_full['Ativo'], name=ticker_symbol, marker_color='tan', boxmean=True))
                fig_b.update_layout(template="plotly_white", height=350)
                st.plotly_chart(fig_b, width='stretch')

            st.markdown("---")
            st.subheader("Comparação Anual e Variação")
            df_y = df_full.groupby('Year')['Ativo'].mean().reset_index()
            df_y['Var %'] = df_y['Ativo'].pct_change() * 100
            def style_neg(v): return f'color: {"red" if v < 0 else "green"}; font-weight: bold;' if pd.notnull(v) else ""
            st.dataframe(df_y.style.applymap(style_neg, subset=['Var %']).format({'Ativo': '{:.2f}', 'Var %': '{:.2f}%'}), width='stretch', hide_index=True)

        with aba_ml:
            with st.spinner("Treinando modelos multivariados..."):
                p_xgb, a_xgb = treinar_xgboost_multi(df_full, dias_predicao)
                p_lstm, a_lstm = treinar_lstm_multi(df_full, dias_predicao)
            st.subheader(f"Probabilidades Direcionais ({dias_predicao} dias)")
            
            # CORREÇÃO: Adicionado eixo Y fantasma para as barras ficarem espessas
            def bar_p(probs, title):
                fig = go.Figure(go.Bar(y=[''], x=[probs[0]], name='Baixa', orientation='h', marker_color='#ffcccb'))
                fig.add_trace(go.Bar(y=[''], x=[probs[1]], name='Neutro', orientation='h', marker_color='#f0f0f0'))
                fig.add_trace(go.Bar(y=[''], x=[probs[2]], name='Alta', orientation='h', marker_color='#d4edda'))
                fig.update_layout(barmode='stack', height=180, title=title, xaxis=dict(range=[0,100], ticksuffix='%'))
                return fig
                
            st.plotly_chart(bar_p(p_xgb, f"XGBoost Multi (Acc: {a_xgb*100:.1f}%)"), width='stretch')
            st.plotly_chart(bar_p(p_lstm, f"LSTM Multi (Acc: {a_lstm*100:.1f}%)"), width='stretch')

        with aba_ia:
            if api_key:
                try:
                    # CORREÇÃO: Utilizando o modelo Gemini 2.0 Flash para otimizar tokens e velocidade
                    genai.configure(api_key=api_key)
                    model_ia = genai.GenerativeModel('gemini-2.0-flash')
                    prompt = f"""
                    Atue como Gestor Sênior. Gere um relatório executivo ESTRITAMENTE técnico para {ticker_symbol}.
                    PROIBIDO: Não use saudações como "Claro", "Aqui está" ou "Olá". Comece direto no conteúdo.
                    
                    ESTRUTURA OBRIGATÓRIA:
                    ### 1. Cenário Macroeconômico e Geopolítico Atual
                    ### 2. Principais Impulsionadores de Preço (Drivers)
                    ### 3. Avaliação dos Modelos Quantitativos
                    - XGBoost: {p_xgb[2]:.1f}% Alta | Acc: {a_xgb*100:.1f}%
                    - LSTM: {p_lstm[2]:.1f}% Alta | Acc: {a_lstm*100:.1f}%
                    ### 4. Perspectivas e Riscos Futuros
                    ### 5. Conclusão Executiva e Sugestão de Alocação (COMPRA, MANUTENÇÃO ou REDUÇÃO)
                    """
                    st.write(model_ia.generate_content(prompt).text)
                except Exception as e: st.error(f"Erro IA: {e}")