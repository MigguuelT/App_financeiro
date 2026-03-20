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
st.title("📈 Análise de Ativos, Predição e Agente de IA")

# --- Barra Lateral ---
st.sidebar.header("Parâmetros da Análise")
ticker_symbol = st.sidebar.text_input("Ticker (ex: HG=F, AAPL, PETR4.SA):", "HG=F").upper()
anos_historico = st.sidebar.number_input("Anos de histórico:", min_value=1, max_value=20, value=10)
dias_predicao = st.sidebar.slider("Dias de predição (Curto Prazo):", 30, 90, 30)

st.sidebar.markdown("---")

# --- Configuração da Chave API ---
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
else:
    api_key = st.sidebar.text_input("Chave API (Google AI Studio):", type="password")

st.sidebar.markdown("---")

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

    df = df.dropna(subset=['Close'])
    df = df.ffill()
    df['Date'] = pd.to_datetime(df['Date'])
    
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # Criando coluna de Ano para o YoY
    df['Year'] = df['Date'].dt.year
    return df

# --- 2. Modelo de Predição ---
def prever_e_avaliar_xgboost(df, dias_futuros):
    df_ml = df[['Date', 'Close']].copy()
    df_ml['Lag_1'] = df_ml['Close'].shift(1)
    df_ml['Lag_2'] = df_ml['Close'].shift(2)
    df_ml['Lag_7'] = df_ml['Close'].shift(7)
    df_ml = df_ml.dropna()
    
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
    
    X_full = df_ml[['Lag_1', 'Lag_2', 'Lag_7']]
    y_full = df_ml['Close']
    modelo_final = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
    modelo_final.fit(X_full, y_full)
    
    predicoes = []
    datas_futuras = [df_ml['Date'].iloc[-1] + timedelta(days=i) for i in range(1, dias_futuros + 1)]
    historico_recente = list(df_ml['Close'].tail(7).values)
    
    for _ in range(dias_futuros):
        x_pred = pd.DataFrame([[historico_recente[-1], historico_recente[-2], historico_recente[-7]]], columns=['Lag_1', 'Lag_2', 'Lag_7'])
        pred_atual = modelo_final.predict(x_pred)[0]
        predicoes.append(pred_atual)
        historico_recente.append(pred_atual)
        
    df_futuro = pd.DataFrame({'Date': datas_futuras, 'Predicao': predicoes})
    df_teste = pd.DataFrame({'Date': test['Date'], 'Real': y_test, 'Predicao_Teste': pred_teste})
    
    return df_futuro, df_teste, metricas

# --- Execução Principal ---
if st.sidebar.button("Analisar Ativo"):
    with st.spinner(f"Coletando dados de {ticker_symbol}..."):
        df = carregar_e_processar_dados(ticker_symbol, anos_historico)
        
        if df.empty:
            st.error("Nenhum dado encontrado. Verifique o Ticker ou sua conexão.")
        else:
            # Organização em Abas (Tabs)
            aba_eda, aba_ml, aba_ia = st.tabs(["📊 Análise Exploratória (EDA)", "🤖 Machine Learning", "🧠 Agente Financeiro"])

            # ==========================================
            # ABA 1: ANÁLISE EXPLORATÓRIA DE DADOS (EDA)
            # ==========================================
            with aba_eda:
                st.subheader(f"Métricas Principais: {ticker_symbol}")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Preço Atual", f"${df['Close'].iloc[-1]:.2f}")
                c2.metric("Média Histórica", f"${df['Close'].mean():.2f}")
                c3.metric("Máxima Histórica", f"${df['High'].max():.2f}")
                c4.metric("Mínima Histórica", f"${df['Low'].min():.2f}")
                
                st.markdown("---")
                
                # Gráfico de Velas (Últimos 6 meses)
                st.subheader("Gráfico de Velas (Últimos 6 Meses)")
                seis_meses_atras = df['Date'].max() - timedelta(days=6*30)
                df_6m = df[df['Date'] >= seis_meses_atras]
                
                fig_candle = go.Figure(data=[go.Candlestick(x=df_6m['Date'], open=df_6m['Open'], high=df_6m['High'], low=df_6m['Low'], close=df_6m['Close'])])
                fig_candle.update_layout(xaxis_rangeslider_visible=False, height=400, template="plotly_white", margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig_candle, use_container_width=True)

                st.markdown("---")
                
                # Distribuição e Boxplot lado a lado
                col_dist, col_box = st.columns(2)
                
                with col_dist:
                    st.subheader("Distribuição de Frequência (Preços)")
                    fig_dist = go.Figure()
                    fig_dist.add_trace(go.Histogram(x=df['Close'], marker_color='lightblue', opacity=0.7, name="Frequência"))
                    
                    media = df['Close'].mean()
                    mediana = df['Close'].median()
                    
                    # Ajuste: Textos em alturas diferentes para evitar sobreposição
                    fig_dist.add_vline(x=media, line_dash="dash", line_color="red", 
                                       annotation_text=f"Média: {media:.2f}", 
                                       annotation_position="top right")
                    
                    fig_dist.add_vline(x=mediana, line_dash="dash", line_color="green", 
                                       annotation_text=f"Mediana: {mediana:.2f}", 
                                       annotation_position="bottom right")
                                       
                    fig_dist.update_layout(height=350, template="plotly_white", margin=dict(l=0, r=0, t=30, b=0))
                    st.plotly_chart(fig_dist, use_container_width=True)

                with col_box:
                    st.subheader("Box Plot de Preços")
                    fig_box = go.Figure()
                    fig_box.add_trace(go.Box(y=df['Close'], name=ticker_symbol, marker_color='tan'))
                    fig_box.update_layout(height=350, template="plotly_white", margin=dict(l=0, r=0, t=30, b=0))
                    st.plotly_chart(fig_box, use_container_width=True)
                
                st.markdown("---")
                
                # Tabela de Comparação YoY (Year over Year)
                st.subheader("Comparação Anual de Preços e Variação (%)")
                df_yoy = df.groupby('Year')['Close'].mean().reset_index()
                df_yoy.columns = ['Ano', 'Preço Médio ($)']
                df_yoy['Variação (%)'] = df_yoy['Preço Médio ($)'].pct_change() * 100
                
                # Formatação visual da tabela
                df_yoy_formatado = df_yoy.copy()
                df_yoy_formatado['Preço Médio ($)'] = df_yoy_formatado['Preço Médio ($)'].apply(lambda x: f"${x:.2f}")
                df_yoy_formatado['Variação (%)'] = df_yoy_formatado['Variação (%)'].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "-")
                
                st.dataframe(df_yoy_formatado, use_container_width=True, hide_index=True)

            # ==========================================
            # ABA 2: MACHINE LEARNING (PREDIÇÃO)
            # ==========================================
            with aba_ml:
                st.subheader("Validação do Modelo (XGBoost)")
                df_pred, df_teste, metricas = prever_e_avaliar_xgboost(df, dias_predicao)
                
                c1, c2, c3 = st.columns(3)
                c1.metric("MAE (Erro Médio Absoluto)", f"${metricas['MAE']:,.3f}")
                c2.metric("RMSE (Raiz do Erro Quadrático)", f"${metricas['RMSE']:,.3f}")
                c3.metric("R² Score (Acurácia)", f"{metricas['R2']:,.3f}")
                
                st.subheader(f"Projeção Futura ({dias_predicao} dias)")
                fig_ml = go.Figure()
                fig_ml.add_trace(go.Scatter(x=df['Date'], y=df['Close'], fill='tozeroy', fillcolor='rgba(210, 180, 140, 0.2)', line=dict(color='tan', width=1.5), name='Histórico Real'))
                fig_ml.add_trace(go.Scatter(x=df_teste['Date'], y=df_teste['Predicao_Teste'], line=dict(color='orange', width=2), name='Validação (Teste)'))
                fig_ml.add_trace(go.Scatter(x=df_pred['Date'], y=df_pred['Predicao'], line=dict(color='red', width=2, dash='dash'), name='Predição Futura'))
                
                fig_ml.update_layout(height=500, template="plotly_white", margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig_ml, use_container_width=True)

            # ==========================================
            # ABA 3: AGENTE FINANCEIRO (IA GEOPOLÍTICA)
            # ==========================================
            with aba_ia:
                st.subheader("Parecer do Agente Geopolítico (Gemini 2.5)")
                if not api_key:
                    st.warning("A Chave da API do Google AI Studio não foi encontrada. Configure-a na barra lateral.")
                else:
                    with st.spinner("O Agente está formulando o cenário macroeconômico atual..."):
                        try:
                            genai.configure(api_key=api_key)
                            # Bug resolvido: Removido o argumento 'tools' que quebrava o SDK
                            agente = genai.GenerativeModel(model_name='gemini-2.5-pro')
                            
                            preco_atual = df['Close'].iloc[-1]
                            variacao = ((preco_atual - df['Close'].iloc[-90]) / df['Close'].iloc[-90]) * 100 if len(df) > 90 else 0
                            
                            prompt = f"""
                            Atue como um Analista Financeiro Sênior e Estrategista Geopolítico.
                            Analise o ativo {ticker_symbol} e seu setor de mercado com base no cenário global mais recente que você conhece.
                            
                            O ativo está cotado hoje a ${preco_atual:.2f} (variação de {variacao:.2f}% em 3 meses). 
                            O nosso modelo preditivo aponta que o preço pode chegar a ${df_pred['Predicao'].iloc[-1]:.2f} em {dias_predicao} dias.
                            
                            Responda de forma analítica e profissional (em português do Brasil):
                            1. Descreva o cenário macroeconômico e geopolítico atual que mais impacta esse ativo (guerras, inflação, supply chain, taxas do FED, etc).
                            2. Aponte os principais riscos de curto prazo.
                            3. Emita um parecer final dizendo se a projeção matemática do modelo preditivo faz sentido frente à realidade do mundo.
                            """
                            resposta = agente.generate_content(prompt)
                            st.write(resposta.text)
                        except Exception as e:
                            st.error(f"Erro na comunicação com a API do Gemini: {e}")