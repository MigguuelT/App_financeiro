import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import google.generativeai as genai

# --- Configuração da Página ---
st.set_page_config(page_title="Análise de Ativos & ML", layout="wide")
st.title("📈 Análise de Ativos, Predição Híbrida e IA")

# --- Barra Lateral ---
st.sidebar.header("Parâmetros da Análise")
ticker_symbol = st.sidebar.text_input("Ticker (ex: HG=F, AAPL, PETR4.SA):", "HG=F").upper()
anos_historico = st.sidebar.number_input("Anos de histórico:", min_value=1, max_value=20, value=10)
dias_predicao = st.sidebar.slider("Dias de predição (Curto Prazo):", 30, 90, 30)

st.sidebar.markdown("---")
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
else:
    api_key = st.sidebar.text_input("Chave API (Google AI Studio):", type="password")
st.sidebar.markdown("---")

# --- 1. Coleta e Pré-processamento Otimizado ---
# O ttl="1d" resolve a lentidão! Mantém os dados no cache por 1 dia.
@st.cache_data(ttl="1d", show_spinner=False)
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
    df['Year'] = df['Date'].dt.year
    return df

# --- 2. Modelos de Machine Learning ---
@st.cache_resource(show_spinner=False) # Cache para não retreinar modelos à toa
def prever_xgboost(df, dias_futuros):
    df_ml = df[['Date', 'Close']].copy()
    df_ml['Lag_1'], df_ml['Lag_2'], df_ml['Lag_7'] = df_ml['Close'].shift(1), df_ml['Close'].shift(2), df_ml['Close'].shift(7)
    df_ml = df_ml.dropna()
    
    train, test = df_ml.iloc[:-dias_futuros], df_ml.iloc[-dias_futuros:]
    X_train, y_train = train[['Lag_1', 'Lag_2', 'Lag_7']], train['Close']
    X_test, y_test = test[['Lag_1', 'Lag_2', 'Lag_7']], test['Close']
    
    modelo_aval = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
    modelo_aval.fit(X_train, y_train)
    pred_teste = modelo_aval.predict(X_test)
    
    metricas = {
        'MAE': mean_absolute_error(y_test, pred_teste),
        'RMSE': np.sqrt(mean_squared_error(y_test, pred_teste)),
        'R2': r2_score(y_test, pred_teste)
    }
    
    modelo_final = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
    modelo_final.fit(df_ml[['Lag_1', 'Lag_2', 'Lag_7']], df_ml['Close'])
    
    predicoes, historico_recente = [], list(df_ml['Close'].tail(7).values)
    for _ in range(dias_futuros):
        pred_atual = modelo_final.predict(pd.DataFrame([[historico_recente[-1], historico_recente[-2], historico_recente[-7]]], columns=['Lag_1', 'Lag_2', 'Lag_7']))[0]
        predicoes.append(pred_atual)
        historico_recente.append(pred_atual)
        
    datas_futuras = [df_ml['Date'].iloc[-1] + timedelta(days=i) for i in range(1, dias_futuros + 1)]
    return pd.DataFrame({'Date': datas_futuras, 'Predicao': predicoes}), metricas

@st.cache_resource(show_spinner=False)
def prever_lstm(df, dias_futuros):
    df_ml = df[['Date', 'Close']].copy().dropna()
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df_ml['Close'].values.reshape(-1, 1))
    
    lookback = 30
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    split_idx = len(X) - dias_futuros
    X_train, y_train, X_test, y_test = X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:]
    
    # Modelo Avaliação
    modelo_aval = Sequential([LSTM(32, input_shape=(lookback, 1)), Dense(1)])
    modelo_aval.compile(optimizer='adam', loss='mean_squared_error')
    modelo_aval.fit(X_train, y_train, batch_size=16, epochs=5, verbose=0)
    
    pred_teste_scaled = modelo_aval.predict(X_test, verbose=0)
    pred_teste = scaler.inverse_transform(pred_teste_scaled).flatten()
    y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    metricas = {
        'MAE': mean_absolute_error(y_test_real, pred_teste),
        'RMSE': np.sqrt(mean_squared_error(y_test_real, pred_teste)),
        'R2': r2_score(y_test_real, pred_teste)
    }
    
    # Modelo Final Futuro
    modelo_final = Sequential([LSTM(32, input_shape=(lookback, 1)), Dense(1)])
    modelo_final.compile(optimizer='adam', loss='mean_squared_error')
    modelo_final.fit(X, y, batch_size=16, epochs=5, verbose=0)
    
    ultimos_dias = scaled_data[-lookback:]
    predicoes_futuras = []
    for _ in range(dias_futuros):
        x_pred = np.reshape(ultimos_dias, (1, lookback, 1))
        pred_atual_scaled = modelo_final.predict(x_pred, verbose=0)
        predicoes_futuras.append(pred_atual_scaled[0, 0])
        ultimos_dias = np.append(ultimos_dias[1:], pred_atual_scaled, axis=0)
        
    predicoes_futuras = scaler.inverse_transform(np.array(predicoes_futuras).reshape(-1, 1)).flatten()
    datas_futuras = [df_ml['Date'].iloc[-1] + timedelta(days=i) for i in range(1, dias_futuros + 1)]
    return pd.DataFrame({'Date': datas_futuras, 'Predicao': predicoes_futuras}), metricas

# --- Execução Principal ---
if st.sidebar.button("Analisar Ativo"):
    with st.spinner(f"Baixando dados do Yahoo Finance... (Isto será rápido nas próximas vezes)"):
        df = carregar_e_processar_dados(ticker_symbol, anos_historico)
        
    if df.empty:
        st.error("Nenhum dado encontrado. Verifique o Ticker ou sua conexão.")
    else:
        aba_eda, aba_ml, aba_ia = st.tabs(["📊 Análise Exploratória (EDA)", "🤖 Machine Learning & Deep Learning", "🧠 Agente Financeiro"])

        # ==========================================
        # ABA 1: ANÁLISE EXPLORATÓRIA (CÓDIGO MANTIDO)
        # ==========================================
        with aba_eda:
            st.subheader(f"Métricas Principais: {ticker_symbol}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Preço Atual", f"${df['Close'].iloc[-1]:.2f}")
            c2.metric("Média", f"${df['Close'].mean():.2f}")
            c3.metric("Máxima", f"${df['High'].max():.2f}")
            c4.metric("Mínima", f"${df['Low'].min():.2f}")
            
            st.markdown("---")
            st.subheader("Gráfico de Velas (Últimos 6 Meses)")
            seis_meses_atras = df['Date'].max() - timedelta(days=6*30)
            df_6m = df[df['Date'] >= seis_meses_atras]
            fig_candle = go.Figure(data=[go.Candlestick(x=df_6m['Date'], open=df_6m['Open'], high=df_6m['High'], low=df_6m['Low'], close=df_6m['Close'])])
            fig_candle.update_layout(xaxis_rangeslider_visible=False, height=400, template="plotly_white", margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_candle, use_container_width=True)

            col_dist, col_box = st.columns(2)
            with col_dist:
                st.subheader("Distribuição de Frequência")
                fig_dist = go.Figure()
                fig_dist.add_trace(go.Histogram(x=df['Close'], marker_color='lightblue', opacity=0.7))
                media, mediana = df['Close'].mean(), df['Close'].median()
                fig_dist.add_vline(x=media, line_dash="dash", line_color="red", annotation_text=f"Média: {media:.2f}", annotation_position="top right")
                fig_dist.add_vline(x=mediana, line_dash="dash", line_color="green", annotation_text=f"Mediana: {mediana:.2f}", annotation_position="bottom right")
                fig_dist.update_layout(height=350, template="plotly_white", margin=dict(l=0, r=0, t=30, b=0), showlegend=False)
                st.plotly_chart(fig_dist, use_container_width=True)

            with col_box:
                st.subheader("Box Plot de Preços")
                fig_box = go.Figure()
                fig_box.add_trace(go.Box(y=df['Close'], marker_color='tan'))
                fig_box.update_layout(height=350, template="plotly_white", margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig_box, use_container_width=True)
            
            st.markdown("---")
            st.subheader("Comparação Anual de Preços e Variação (%)")
            df_yoy = df.groupby('Year')['Close'].mean().reset_index()
            df_yoy.columns = ['Ano', 'Preço Médio ($)']
            df_yoy['Variação (%)'] = df_yoy['Preço Médio ($)'].pct_change() * 100
            df_yoy_formatado = df_yoy.copy()
            df_yoy_formatado['Preço Médio ($)'] = df_yoy_formatado['Preço Médio ($)'].apply(lambda x: f"${x:.2f}")
            df_yoy_formatado['Variação (%)'] = df_yoy_formatado['Variação (%)'].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "-")
            st.dataframe(df_yoy_formatado, use_container_width=True, hide_index=True)

        # ==========================================
        # ABA 2: MACHINE LEARNING & DEEP LEARNING
        # ==========================================
        with aba_ml:
            with st.spinner("Treinando modelos XGBoost e Rede Neural LSTM... Isso leva alguns segundos."):
                df_xgb, met_xgb = prever_xgboost(df, dias_predicao)
                df_lstm, met_lstm = prever_lstm(df, dias_predicao)
            
            # Tabela de Comparação de Métricas
            st.subheader("🏆 Comparação de Desempenho dos Modelos")
            df_metricas = pd.DataFrame({
                'Modelo': ['XGBoost (Árvores)', 'LSTM (Rede Neural)'],
                'MAE (Menor é melhor)': [f"${met_xgb['MAE']:.3f}", f"${met_lstm['MAE']:.3f}"],
                'RMSE (Menor é melhor)': [f"${met_xgb['RMSE']:.3f}", f"${met_lstm['RMSE']:.3f}"],
                'R² Score (Próximo a 1 é melhor)': [f"{met_xgb['R2']:.3f}", f"{met_lstm['R2']:.3f}"]
            })
            st.dataframe(df_metricas, use_container_width=True, hide_index=True)
            
            # Gráfico com ambas as projeções
            st.subheader(f"Projeção Futura Comparativa ({dias_predicao} dias)")
            fig_ml = go.Figure()
            # Mostrando apenas os últimos 2 anos no gráfico para o zoom ficar bom
            dois_anos = df[df['Date'] >= (df['Date'].max() - timedelta(days=730))]
            
            fig_ml.add_trace(go.Scatter(x=dois_anos['Date'], y=dois_anos['Close'], line=dict(color='gray', width=1.5), name='Histórico Real'))
            fig_ml.add_trace(go.Scatter(x=df_xgb['Date'], y=df_xgb['Predicao'], line=dict(color='red', width=2, dash='dash'), name='Predição XGBoost'))
            fig_ml.add_trace(go.Scatter(x=df_lstm['Date'], y=df_lstm['Predicao'], line=dict(color='blue', width=2, dash='dot'), name='Predição LSTM'))
            
            fig_ml.update_layout(
                height=500, 
                template="plotly_white", 
                margin=dict(l=0, r=0, t=40, b=0), 
                legend=dict(
                    orientation="h", 
                    yanchor="bottom", 
                    y=1.02, 
                    xanchor="left", # <-- Mudamos de "right" para "left"
                    x=0             # <-- Mudamos de 1 para 0 (início do eixo X)
                )
            )
            st.plotly_chart(fig_ml, use_container_width=True)

        # ==========================================
        # ABA 3: AGENTE FINANCEIRO (RELATÓRIO INSTITUCIONAL)
        # ==========================================
        with aba_ia:
            st.subheader("Relatório Executivo: IA e Quantitativo")
            if not api_key:
                st.warning("Configure a Chave da API do Google AI Studio na barra lateral.")
            else:
                with st.spinner("Sintetizando dados quantitativos, coletando notícias frescas da semana e formulando o relatório..."):
                    try:
                        genai.configure(api_key=api_key)
                        agente = genai.GenerativeModel(model_name='gemini-2.5-pro')
                        
                        # Coletando dados estatísticos
                        preco_atual = df['Close'].iloc[-1]
                        variacao = ((preco_atual - df['Close'].iloc[-90]) / df['Close'].iloc[-90]) * 100 if len(df) > 90 else 0
                        volatilidade = df['Close'].tail(30).std()
                        
                        # COLETANDO NOTÍCIAS EM TEMPO REAL (Solução Pytonica Robusta)
                        ativo_yf = yf.Ticker(ticker_symbol)
                        noticias_brutas = ativo_yf.news
                        
                        # Formata as 5 notícias mais recentes (se existirem) para enviar ao LLM
                        if noticias_brutas:
                            manchetes = "\n".join([f"- {n.get('title', 'Sem título')} (Fonte: {n.get('publisher', 'Desconhecida')})" for n in noticias_brutas[:5]])
                        else:
                            manchetes = "Nenhuma manchete específica de grande impacto encontrada nas últimas horas para este ticker diretamente."

                        prompt = f"""
                        Atue como um Analista Quantitativo Sênior e Estrategista Macroeconômico de um fundo de investimentos tier-1.
                        Sua tarefa é redigir um relatório analítico executivo e profissional sobre o ativo {ticker_symbol}.
                        
                        DADOS DE MERCADO ATUAIS:
                        - Preço Atual: ${preco_atual:.2f}
                        - Variação (3 Meses): {variacao:.2f}%
                        - Volatilidade (Desvio Padrão 30d): ${volatilidade:.2f}
                        
                        MANCHETES E EVENTOS DESTA SEMANA SOBRE O ATIVO:
                        {manchetes}
                        
                        DADOS DOS MODELOS PREDITIVOS (Projeção para {dias_predicao} dias):
                        - Previsão XGBoost (Machine Learning): ${df_xgb['Predicao'].iloc[-1]:.2f} | Confiança do Modelo -> MAE: ${met_xgb['MAE']:.2f}, R²: {met_xgb['R2']:.2f}
                        - Previsão LSTM (Deep Learning): ${df_lstm['Predicao'].iloc[-1]:.2f} | Confiança do Modelo -> MAE: ${met_lstm['MAE']:.2f}, R²: {met_lstm['R2']:.2f}
                        *(Nota: R² próximo de 1 indica alta confiabilidade. MAE menor indica menor erro médio).*
                        
                        Com base nesses dados quantitativos, nas manchetes frescas fornecidas e no seu amplo conhecimento do cenário global, gere um relatório formatado em Markdown com os seguintes tópicos obrigatórios:
                        
                        ### 1. Cenário Macroeconômico e Geopolítico Atual
                        Descreva o contexto global que afeta este ativo especificamente.
                        
                        ### 2. Principais Impulsionadores de Preço (Drivers)
                        Liste e explique os fatores centrais que estão movendo o preço do ativo no momento.
                        
                        ### 3. Impacto das Notícias e Eventos da Semana
                        Analise rigorosamente como as manchetes recentes listadas acima (ou os fatos geopolíticos dos últimos dias) estão influenciando o sentimento do mercado e ditando a variação atual dos preços.
                        
                        ### 4. Avaliação dos Modelos Quantitativos
                        Analise o desempenho do XGBoost vs LSTM com base no MAE e R². Os modelos concordam? A projeção matemática faz sentido ou está enviesada frente aos eventos geopolíticos da semana?
                        
                        ### 5. Perspectivas e Riscos Futuros
                        Projete o cenário esperado para os próximos {dias_predicao} dias. Inclua riscos de cauda (eventos inesperados que podem invalidar as predições).
                        
                        ### 6. Conclusão Executiva
                        Um parágrafo final resumindo a tese.
                        
                        O tom deve ser estritamente institucional, objetivo, sofisticado e imparcial.
                        """
                        
                        resposta = agente.generate_content(prompt)
                        st.write(resposta.text)
                        
                        # Disclaimer
                        st.markdown("---")
                        st.caption("⚠️ **Aviso Legal:** Este relatório é gerado por Inteligência Artificial a partir de modelos estatísticos. Não constitui recomendação de compra, venda ou indicação de investimento. O mercado financeiro é volátil e os dados do passado não garantem rentabilidade futura.")
                        
                    except Exception as e:
                        st.error(f"Erro na comunicação com a API: {e}")