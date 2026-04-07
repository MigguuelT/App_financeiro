# ==========================================================================================
# PROJETO: ANALISADOR QUANTITATIVO MULTIVARIADO (PRODUÇÃO + BOTÃO DOWNLOAD DADOS / APROVADO)
# ==========================================================================================

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
from sklearn.model_selection import TimeSeriesSplit

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
ticker_symbol = st.sidebar.text_input("Ativo Principal (ex: QQQ, IAU):", "QQQ").upper()
ticker_macro_input = st.sidebar.text_input("Indexador Macro (ex: ^TNX, UUP):", "^TNX").upper()
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
def carregar_dados_completos(ticker_principal, ticker_macro, anos):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=anos * 365)
    
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
            data = data.rename(columns={ticker_principal: 'Ativo', ticker_macro: 'Macro'})
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
        print(f"Erro no pipeline de dados: {e}")
        return pd.DataFrame(), "", pd.DataFrame()

@st.cache_data(show_spinner=False)
def preparar_dados_csv(df, dias_futuros):
    """Gera o dataset consolidado com todas as features para download."""
    df_ml = df.copy()
    
    for lag in [1, 5, 21]:
        df_ml[f'Lag_Ativo_{lag}'] = df_ml['Ret_Ativo'].shift(lag)
        df_ml[f'Lag_Macro_{lag}'] = df_ml['Ret_Macro'].shift(lag)
        
    df_ml['Dist_SMA_200'] = (df_ml['Ativo'] - df_ml['Ativo'].rolling(200).mean()) / df_ml['Ativo'].rolling(200).mean()
    desvio_padrao_20 = df_ml['Ativo'].rolling(20).std()
    media_20 = df_ml['Ativo'].rolling(20).mean()
    df_ml['Dist_Banda_Sup'] = (df_ml['Ativo'] - (media_20 + (desvio_padrao_20 * 2))) / (media_20 + (desvio_padrao_20 * 2))
    df_ml['Dist_Banda_Inf'] = (df_ml['Ativo'] - (media_20 - (desvio_padrao_20 * 2))) / (media_20 - (desvio_padrao_20 * 2))
    df_ml['Volatilidade_Ativo'] = df_ml['Ret_Ativo'].rolling(21).std()
    
    ret_futuro = (df_ml['Ativo'].shift(-dias_futuros) - df_ml['Ativo']) / df_ml['Ativo']
    limite = df_ml['Ret_Ativo'].std() * np.sqrt(dias_futuros) * 0.5
    
    df_ml['Target'] = np.select([(ret_futuro > limite), (ret_futuro < -limite)], [2, 0], default=1)
    df_ml.loc[ret_futuro.isna(), 'Target'] = np.nan
    
    return df_ml.dropna().to_csv(index=False).encode('utf-8')

@st.cache_resource(show_spinner=False)
def treinar_xgboost_multi(df, dias_futuros):
    df_ml = df.copy()
    
    for lag in [1, 5, 21]:
        df_ml[f'Lag_Ativo_{lag}'] = df_ml['Ret_Ativo'].shift(lag)
        df_ml[f'Lag_Macro_{lag}'] = df_ml['Ret_Macro'].shift(lag)
        
    df_ml['Dist_SMA_200'] = (df_ml['Ativo'] - df_ml['Ativo'].rolling(200).mean()) / df_ml['Ativo'].rolling(200).mean()
    desvio_padrao_20 = df_ml['Ativo'].rolling(20).std()
    media_20 = df_ml['Ativo'].rolling(20).mean()
    df_ml['Dist_Banda_Sup'] = (df_ml['Ativo'] - (media_20 + (desvio_padrao_20 * 2))) / (media_20 + (desvio_padrao_20 * 2))
    df_ml['Dist_Banda_Inf'] = (df_ml['Ativo'] - (media_20 - (desvio_padrao_20 * 2))) / (media_20 - (desvio_padrao_20 * 2))
    df_ml['Volatilidade_Ativo'] = df_ml['Ret_Ativo'].rolling(21).std()
    
    features = [c for c in df_ml.columns if 'Lag' in c or 'Correl' in c or 'Dist' in c or 'Volatilidade' in c]
    
    df_ml_temp = df_ml.dropna(subset=features)
    vetor_hoje = df_ml_temp.iloc[-1:][features].copy()
    
    ret_futuro = (df_ml_temp['Ativo'].shift(-dias_futuros) - df_ml_temp['Ativo']) / df_ml_temp['Ativo']
    limite = df_ml_temp['Ret_Ativo'].std() * np.sqrt(dias_futuros) * 0.5
    
    df_ml_temp['Target'] = np.select([(ret_futuro > limite), (ret_futuro < -limite)], [2, 0], default=1)
    df_ml_temp.loc[ret_futuro.isna(), 'Target'] = np.nan
    
    df_treino = df_ml_temp.dropna()
    X = df_treino[features]
    y = df_treino['Target'].astype(int)
    
    tscv = TimeSeriesSplit(n_splits=5)
    acuracias_cv = []
    model_cv = xgb.XGBClassifier(objective='multi:softprob', num_class=3, n_estimators=100)
    
    for train_index, test_index in tscv.split(X):
        model_cv.fit(X.iloc[train_index], y.iloc[train_index])
        preds = model_cv.predict(X.iloc[test_index])
        acuracias_cv.append(accuracy_score(y.iloc[test_index], preds))
        
    acuracia_final_robusta = np.mean(acuracias_cv)
    
    model_final = xgb.XGBClassifier(objective='multi:softprob', num_class=3, n_estimators=100)
    model_final.fit(X, y)
    prob_hoje = model_final.predict_proba(vetor_hoje)[0] * 100
    
    return prob_hoje, acuracia_final_robusta

@st.cache_resource(show_spinner=False)
def treinar_lstm_multi(df, dias_futuros):
    df_ml = df.copy()
    df_ml['Volatilidade_Ativo'] = df_ml['Ret_Ativo'].rolling(21).std()
    df_ml = df_ml.dropna()
    
    scaler = StandardScaler()
    scaled_f = scaler.fit_transform(df_ml[['Ret_Ativo', 'Ret_Macro', 'Volatilidade_Ativo']])
    
    ret_f = (df_ml['Ativo'].shift(-dias_futuros) - df_ml['Ativo']) / df_ml['Ativo']
    limite = df_ml['Ret_Ativo'].std() * np.sqrt(dias_futuros) * 0.5
    targets = np.select([(ret_f > limite), (ret_f < -limite)], [2, 0], default=1)
    
    lookback = 60
    X, y = [], []
    
    for i in range(lookback, len(scaled_f) - dias_futuros):
        X.append(scaled_f[i-lookback:i])
        y.append(targets[i])
        
    X, y = np.array(X), np.array(y)
    split = int(len(X) * 0.8)
    
    model = Sequential([
        LSTM(32, input_shape=(lookback, 3)), 
        Dropout(0.2),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X[:split], y[:split], validation_data=(X[split:], y[split:]), epochs=30, batch_size=32, verbose=0, callbacks=[early_stop])
    
    acc = accuracy_score(y[split:], np.argmax(model.predict(X[split:], verbose=0), axis=1))
    prob_hoje = model.predict(np.reshape(scaled_f[-lookback:], (1, lookback, 3)), verbose=0)[0] * 100
    
    return prob_hoje, acc

# ==============================================================================
# FRONT-END (INTERFACE E DASHBOARD)
# ==============================================================================

if st.sidebar.button("Analisar Ativo"):
    df_full, ticker_m, df_ohlc = carregar_dados_completos(ticker_symbol, ticker_macro_input, anos_historico)
    
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
                x=df_ohlc.index, open=df_ohlc['Open'], high=df_ohlc['High'], low=df_ohlc['Low'], close=df_ohlc['Close'], name=ticker_symbol
            )])
            fig_v.update_layout(
                xaxis_rangeslider_visible=False, height=400, template="plotly_white", margin=dict(l=0, r=0, t=30, b=0),
                xaxis_title="Data", yaxis_title=f"Preço ({moeda})"
            )
            st.plotly_chart(fig_v, use_container_width=True)

            col_d, col_b = st.columns(2, gap="large")
            with col_d:
                st.subheader("Distribuição de Frequência")
                fig_d = go.Figure(go.Histogram(x=df_full['Ativo'], marker_color='lightblue', opacity=0.7))
                m_avg, m_med = df_full['Ativo'].mean(), df_full['Ativo'].median()
                
                fig_d.add_vline(x=m_avg, line_dash="dash", line_color="red", annotation_text=f"Média: {moeda} {m_avg:.2f}", annotation_position="top right", annotation_yshift=20, annotation_bgcolor="rgba(255, 0, 0, 0.1)")
                fig_d.add_vline(x=m_med, line_dash="dash", line_color="green", annotation_text=f"Mediana: {moeda} {m_med:.2f}", annotation_position="top right", annotation_yshift=0, annotation_bgcolor="rgba(0, 255, 0, 0.1)")
                
                fig_d.update_layout(
                    template="plotly_white", height=350,
                    xaxis_title=f"Preço de Fechamento ({moeda})", yaxis_title="Frequência (Dias)"
                )
                st.plotly_chart(fig_d, use_container_width=True)
                
            with col_b:
                st.subheader("Box Plot de Preços")
                fig_b = go.Figure(go.Box(y=df_full['Ativo'], name=ticker_symbol, marker_color='tan', boxmean=True))
                
                fig_b.update_layout(
                    template="plotly_white", height=350,
                    yaxis_title=f"Distribuição de Preço ({moeda})"
                )
                st.plotly_chart(fig_b, use_container_width=True)

            st.markdown("---")
            st.subheader("Comparação Anual e Variação")
            df_y = df_full.groupby('Year')['Ativo'].mean().reset_index()
            nome_col_preco = f'Preço Médio ({moeda})'
            df_y.columns = ['Ano', nome_col_preco]
            df_y['Variação (%)'] = df_y[nome_col_preco].pct_change() * 100
            
            def style_neg(v): 
                return f'color: {"red" if v < 0 else "green"}; font-weight: bold;' if pd.notnull(v) else ""
            
            st.dataframe(
                df_y.style.map(style_neg, subset=['Variação (%)']).format({
                    nome_col_preco: '{:.2f}', 'Variação (%)': '{:.2f}%'
                }), 
                use_container_width=True, hide_index=True
            )

        with aba_ml:
            st.subheader("Contexto de Tendência (SMAs)")
            df_sma = df_full.tail(500).copy()
            
            fig_sma = go.Figure()
            fig_sma.add_trace(go.Scatter(x=df_sma['Date'], y=df_sma['Ativo'], mode='lines', name='Preço Real', line=dict(color='gray', width=1.5)))
            
            for periodo, cor in zip([20, 50, 200], ['blue', 'orange', 'red']):
                sma_valores = df_sma['Ativo'].rolling(window=periodo).mean()
                linha_dash = 'dash' if periodo == 200 else 'solid'
                espessura = 2 if periodo == 200 else 1.2
                fig_sma.add_trace(go.Scatter(x=df_sma['Date'], y=sma_valores, mode='lines', name=f'SMA {periodo}', line=dict(color=cor, width=espessura, dash=linha_dash)))
                
            fig_sma.update_layout(
                height=400, template="plotly_white", margin=dict(l=0, r=0, t=30, b=0),
                xaxis_title="Data", yaxis_title=f"Preço ({moeda})",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
            )
            st.plotly_chart(fig_sma, use_container_width=True)
            
            st.markdown("---")

            with st.spinner("Treinando modelos multivariados (Walk-Forward Validation)..."):
                p_xgb, a_xgb = treinar_xgboost_multi(df_full, dias_predicao)
                p_lstm, a_lstm = treinar_lstm_multi(df_full, dias_predicao)
            
            st.subheader(f"Probabilidades Direcionais ({dias_predicao} dias)")
            
            def bar_p(probs, title):
                fig = go.Figure()
                fig.add_trace(go.Bar(y=[''], x=[probs[0]], name='Baixa', orientation='h', marker_color='#ffcccb', width=0.6))
                fig.add_trace(go.Bar(y=[''], x=[probs[1]], name='Neutro', orientation='h', marker_color='#f0f0f0', width=0.6))
                fig.add_trace(go.Bar(y=[''], x=[probs[2]], name='Alta', orientation='h', marker_color='#d4edda', width=0.6))
                
                fig.update_layout(
                    barmode='stack', height=130, title=title, 
                    xaxis=dict(range=[0,100], ticksuffix='%'), 
                    yaxis=dict(showticklabels=False), 
                    margin=dict(l=0, r=0, t=30, b=10)
                )
                return fig
                
            st.plotly_chart(bar_p(p_xgb, f"XGBoost Multi (Acc: {a_xgb*100:.1f}%)"), use_container_width=True)
            st.plotly_chart(bar_p(p_lstm, f"LSTM Multi (Acc: {a_lstm*100:.1f}%)"), use_container_width=True)

            # --- BOTÃO DE EXPORTAÇÃO DE DADOS ---
            st.markdown("---")
            csv_dados = preparar_dados_csv(df_full, dias_predicao)
            st.download_button(
                label="💾 Baixar Dados de Treinamento (CSV)",
                data=csv_dados,
                file_name=f"{ticker_symbol}_features_quant.csv",
                mime="text/csv",
                use_container_width=True
            )

        with aba_ia:
            if api_key:
                try:
                    genai.configure(api_key=api_key)
                    model_ia = genai.GenerativeModel('gemma-3-27b-it')
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
