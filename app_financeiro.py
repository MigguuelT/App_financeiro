import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import google.generativeai as genai

# --- Configuração da Página ---
st.set_page_config(page_title="Análise de Ativos & ML", layout="wide")
st.title("📈 Análise de Ativos, Predição de Curto Prazo e Agente de IA")

# --- Barra Lateral ---
st.sidebar.header("Parâmetros da Análise")
ticker_symbol = st.sidebar.text_input("Ticker (ex: HG=F, AAPL, PETR4.SA):", "HG=F").upper()
anos_historico = st.sidebar.number_input("Anos de histórico:", min_value=1, max_value=20, value=10)
dias_predicao = st.sidebar.slider("Dias de predição (Curto Prazo):", 30, 90, 30)

st.sidebar.markdown("---")
api_key = st.sidebar.text_input("Chave API (Google AI Studio):", type="password")

# --- 1. Coleta e Pré-processamento ---
@st.cache_data
def carregar_e_processar_dados(ticker, anos):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=anos * 365)
    
    df = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), progress=False)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    
    if df.empty: return df

    # Limpeza
    df = df.dropna(subset=['Close'])
    df = df.ffill()
    df['Date'] = pd.to_datetime(df['Date'])
    
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    return df

# --- 2. Modelo de Predição e Avaliação ---
def prever_e_avaliar_xgboost(df, dias_futuros):
    df_ml = df[['Date', 'Close']].copy()
    
    # Engenharia de Recursos
    df_ml['Lag_1'] = df_ml['Close'].shift(1)
    df_ml['Lag_2'] = df_ml['Close'].shift(2)
    df_ml['Lag_7'] = df_ml['Close'].shift(7)
    df_ml = df_ml.dropna()
    
    # --- FASE 1: Avaliação (Train/Test Split) ---
    dias_teste = dias_futuros
    train = df_ml.iloc[:-dias_teste]
    test = df_ml.iloc[-dias_teste:]
    
    X_train, y_train = train[['Lag_1', 'Lag_2', 'Lag_7']], train['Close']
    X_test, y_test = test[['Lag_1', 'Lag_2', 'Lag_7']], test['Close']
    
    modelo_aval = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
    modelo_aval.fit(X_train, y_train)
    pred_teste = modelo_aval.predict(X_test)
    
    mae = mean_absolute_error(y_test, pred_teste)
    rmse = np.sqrt(mean_squared_error(y_test, pred_teste))
    r2 = r2_score(y_test, pred_teste)
    
    metricas = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
    
    # --- FASE 2: Predição Futura (Treinando com tudo) ---
    X_full = df_ml[['Lag_1', 'Lag_2', 'Lag_7']]
    y_full = df_ml['Close']
    
    modelo_final = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
    modelo_final.fit(X_full, y_full)
    
    predicoes = []
    datas_futuras = [df_ml['Date'].iloc[-1] + timedelta(days=i) for i in range(1, dias_futuros + 1)]
    historico_recente = list(df_ml['Close'].tail(7).values)
    
    for _ in range(dias_futuros):
        x_pred = pd.DataFrame([[historico_recente[-1], historico_recente[-2], historico_recente[-7]]], 
                              columns=['Lag_1', 'Lag_2', 'Lag_7'])
        pred_atual = modelo_final.predict(x_pred)[0]
        predicoes.append(pred_atual)
        historico_recente.append(pred_atual)
        
    df_futuro = pd.DataFrame({'Date': datas_futuras, 'Predicao': predicoes})
    df_teste = pd.DataFrame({'Date': test['Date'], 'Real': y_test, 'Predicao_Teste': pred_teste})
    
    return df_futuro, df_teste, metricas

# --- Execução Principal ---
if st.sidebar.button("Analisar Ativo"):
    with st.spinner(f"Coletando dados e processando {ticker_symbol}..."):
        df = carregar_e_processar_dados(ticker_symbol, anos_historico)
        
        if df.empty:
            st.error("Nenhum dado encontrado. Verifique o Ticker.")
        else:
            st.subheader("📊 Estatísticas e Validação do Modelo")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Preço Atual", f"${df['Close'].iloc[-1]:.2f}")
            
            df_pred, df_teste, metricas = prever_e_avaliar_xgboost(df, dias_predicao)
            
            col2.metric("MAE (Erro Médio)", f"${metricas['MAE']:,.3f}")
            col3.metric("RMSE (Raiz do Erro)", f"${metricas['RMSE']:,.3f}")
            col4.metric("R² Score", f"{metricas['R2']:,.3f}")
            
            st.subheader("📈 Análise de Preços e Projeções Futuras")
            fig = go.Figure()
            
            # Histórico
            fig.add_trace(go.Scatter(
                x=df['Date'], y=df['Close'], fill='tozeroy', 
                fillcolor='rgba(210, 180, 140, 0.2)', line=dict(color='tan', width=1.5), name='Histórico'
            ))
            
            # Predição Teste (Validação)
            fig.add_trace(go.Scatter(
                x=df_teste['Date'], y=df_teste['Predicao_Teste'],
                line=dict(color='orange', width=2), name='Validação do Modelo (Teste)'
            ))
            
            # Predição Futura
            fig.add_trace(go.Scatter(
                x=df_pred['Date'], y=df_pred['Predicao'],
                line=dict(color='red', width=2, dash='dash'), name='Predição Futura (XGBoost)'
            ))
            
            fig.update_layout(height=500, margin=dict(l=0, r=0, t=30, b=0), template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

            # --- AGENTE FINANCEIRO ---
            st.subheader("🤖 Agente Financeiro: Cenário e Geopolítica")
            
            # --- Configuração da Chave API ---
			# Tenta buscar a chave nos secrets do Streamlit (Local ou Cloud)
			if "GEMINI_API_KEY" in st.secrets:
			    api_key = st.secrets["GEMINI_API_KEY"]
			else:
			    # Plano B: Input manual na barra lateral caso o secret não esteja configurado
			    api_key = st.sidebar.text_input("Chave API (Google AI Studio):", type="password")
			    st.sidebar.markdown("---")
            else:
                with st.spinner("Buscando manchetes em tempo real e projetando o cenário..."):
                    try:
                        genai.configure(api_key=api_key)
                        agente = genai.GenerativeModel(model_name='gemini-2.5-pro', tools='google_search_retrieval')
                        
                        preco_atual = df['Close'].iloc[-1]
                        variacao = ((preco_atual - df['Close'].iloc[-90]) / df['Close'].iloc[-90]) * 100 if len(df) > 90 else 0
                        
                        prompt = f"""
                        Busque as notícias geopolíticas e macroeconômicas mais recentes de hoje sobre o ativo {ticker_symbol} e seu setor.
                        O ativo está cotado a ${preco_atual:.2f} (variação de {variacao:.2f}% em 3 meses). 
                        O modelo de machine learning previu que o preço chegará a ${df_pred['Predicao'].iloc[-1]:.2f} no dia {df_pred['Date'].iloc[-1].strftime('%d/%m/%Y')}.
                        
                        Com base na sua busca, aja como um Agente Financeiro sênior:
                        1. Descreva o cenário atual.
                        2. Identifique os riscos e tendências geopolíticas de curto prazo.
                        3. Emita um parecer sobre a viabilidade da projeção do modelo preditivo frente às notícias atuais.
                        """
                        resposta = agente.generate_content(prompt)
                        st.write(resposta.text)
                    except Exception as e:
                        st.error(f"Erro ao processar o agente: {e}")