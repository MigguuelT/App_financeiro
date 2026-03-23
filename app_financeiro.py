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
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
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
@st.cache_resource(show_spinner=False)
def prever_xgboost(df, dias_futuros):
    df_ml = df[['Date', 'Close']].copy()
    
    # 1. Feature Engineering Avançado: Criando Indicadores Técnicos
    # Lags de preço
    lags = [1, 2, 3, 5, 7, 10]
    for lag in lags:
        df_ml[f'Lag_{lag}'] = df_ml['Close'].shift(lag)
        
    # Média Móvel Simples (SMA) de 10 e 20 dias (Tendência)
    df_ml['SMA_10'] = df_ml['Close'].rolling(window=10).mean()
    df_ml['SMA_20'] = df_ml['Close'].rolling(window=20).mean()
    
    # Índice de Força Relativa (RSI) de 14 dias (Momentum / Sobrevenda / Sobrecompra)
    delta = df_ml['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_ml['RSI_14'] = 100 - (100 / (1 + rs))
        
    df_ml = df_ml.dropna()
    
    # Selecionando as novas features para o treinamento
    features = [f'Lag_{lag}' for lag in lags] + ['SMA_10', 'SMA_20', 'RSI_14']
    
    train, test = df_ml.iloc[:-dias_futuros], df_ml.iloc[-dias_futuros:]
    X_train, y_train = train[features], train['Close']
    X_test, y_test = test[features], test['Close']
    
    parametros = {
        'objective': 'reg:squarederror',
        'n_estimators': 150,
        'learning_rate': 0.05,
        'max_depth': 4,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }
    
    # Treino e Avaliação
    modelo_aval = xgb.XGBRegressor(**parametros)
    modelo_aval.fit(X_train, y_train)
    pred_teste = modelo_aval.predict(X_test)
    
    metricas = {
        'MAE': mean_absolute_error(y_test, pred_teste),
        'RMSE': np.sqrt(mean_squared_error(y_test, pred_teste)),
        'R2': r2_score(y_test, pred_teste)
    }
    
    # Modelo Final Futuro
    modelo_final = xgb.XGBRegressor(**parametros)
    modelo_final.fit(df_ml[features], df_ml['Close'])
    
    # Predição iterativa (Atualizando as features a cada passo para o futuro)
    predicoes = []
    historico_recente = list(df_ml['Close'].values)
    
    for i in range(dias_futuros):
        hist_series = pd.Series(historico_recente)
        
        # Recalculando os indicadores para o "dia de amanhã"
        x_pred_dict = {f'Lag_{lag}': [historico_recente[-lag]] for lag in lags}
        x_pred_dict['SMA_10'] = [hist_series.tail(10).mean()]
        x_pred_dict['SMA_20'] = [hist_series.tail(20).mean()]
        
        # Recalculando o RSI de forma simplificada para a predição
        delta_fut = hist_series.diff()
        gain_fut = (delta_fut.where(delta_fut > 0, 0)).tail(14).mean()
        loss_fut = (-delta_fut.where(delta_fut < 0, 0)).tail(14).mean()
        rs_fut = gain_fut / loss_fut if loss_fut != 0 else 0
        x_pred_dict['RSI_14'] = [100 - (100 / (1 + rs_fut))]
        
        x_pred = pd.DataFrame(x_pred_dict)
        pred_atual = modelo_final.predict(x_pred)[0]
        
        predicoes.append(pred_atual)
        historico_recente.append(pred_atual)
        
    datas_futuras = [df_ml['Date'].iloc[-1] + timedelta(days=i) for i in range(1, dias_futuros + 1)]
    return pd.DataFrame({'Date': datas_futuras, 'Predicao': predicoes}), metricas

@st.cache_resource(show_spinner=False)
def prever_lstm(df, dias_futuros):
    df_ml = df[['Date', 'Close']].copy().dropna()
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df_ml['Close'].values.reshape(-1, 1))
    
    # 1. Reduzimos a memória para 21 dias (1 mês útil). 
    # Em ações ruidosas como GOOG, olhar 60 dias de preços crus confunde o modelo.
    lookback = 21 
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    split_idx = len(X) - dias_futuros
    X_train, y_train, X_test, y_test = X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:]
    
    def construir_modelo():
        modelo = Sequential()
        # 2. Arquitetura mais enxuta (32 neurônios) para não decorar o ruído
        modelo.add(LSTM(units=32, return_sequences=False, input_shape=(lookback, 1)))
        modelo.add(Dropout(0.2)) # Mantemos o dropout para forçar a generalização
        modelo.add(Dense(units=1)) # Ativação linear (sem sigmoid) para permitir novas máximas históricas
        
        # Usamos uma taxa de aprendizado ligeiramente menor para estabilizar o treino
        otimizador = tf.keras.optimizers.Adam(learning_rate=0.001)
        modelo.compile(optimizer=otimizador, loss='mean_squared_error')
        return modelo
    
    # 3. Early Stopping mais rigoroso (para rápido se a perda de validação piorar)
    early_stop = EarlyStopping(monitor='loss', patience=4, restore_best_weights=True)
    
    modelo_aval = construir_modelo()
    modelo_aval.fit(X_train, y_train, batch_size=16, epochs=40, verbose=0, callbacks=[early_stop])
    
    pred_teste_scaled = modelo_aval.predict(X_test, verbose=0)
    pred_teste = scaler.inverse_transform(pred_teste_scaled).flatten()
    y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    metricas = {
        'MAE': mean_absolute_error(y_test_real, pred_teste),
        'RMSE': np.sqrt(mean_squared_error(y_test_real, pred_teste)),
        'R2': r2_score(y_test_real, pred_teste)
    }
    
    modelo_final = construir_modelo()
    modelo_final.fit(X, y, batch_size=16, epochs=40, verbose=0, callbacks=[early_stop])
    
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
    with st.spinner("Baixando dados do Yahoo Finance..."):
        df = carregar_e_processar_dados(ticker_symbol, anos_historico)
        
    if df.empty:
        st.error("Nenhum dado encontrado. Verifique o Ticker ou sua conexão.")
    else:
        aba_eda, aba_ml, aba_ia = st.tabs(["📊 Análise Exploratória (EDA)", "🤖 Machine Learning & Deep Learning", "🧠 Agente Financeiro"])

        # ==========================================
        # ABA 1: ANÁLISE EXPLORATÓRIA
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
                fig_box.add_trace(go.Box(
                    y=df['Close'], 
                    name=ticker_symbol,           # Substitui "trace 0" pelo nome do ativo no eixo X
                    marker_color='tan',
                    boxpoints='outliers',         # Garante que pontos anômalos (outliers) fiquem visíveis
                    yhoverformat="$,.2f"          # Formata todas as métricas do boxplot com $ e 2 casas decimais
                ))
                
                fig_box.update_layout(
                    height=350, 
                    template="plotly_white", 
                    margin=dict(l=0, r=0, t=30, b=0),
                    yaxis_title="Preço de Fechamento" # Título lateral para o eixo Y
                )
                
                # Formata os números do próprio eixo Y com cifrão
                fig_box.update_yaxes(tickprefix="$", tickformat=".2f") 
                
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
            with st.spinner("Treinando modelos XGBoost e Rede Neural LSTM Profunda... (Aguarde alguns segundos)"):
                df_xgb, met_xgb = prever_xgboost(df, dias_predicao)
                df_lstm, met_lstm = prever_lstm(df, dias_predicao)
            
            st.subheader("🏆 Comparação de Desempenho dos Modelos")
            df_metricas = pd.DataFrame({
                'Modelo': ['XGBoost Tunado', 'LSTM Otimizada'],
                'MAE (Menor é melhor)': [f"${met_xgb['MAE']:.3f}", f"${met_lstm['MAE']:.3f}"],
                'RMSE (Menor é melhor)': [f"${met_xgb['RMSE']:.3f}", f"${met_lstm['RMSE']:.3f}"],
                'R² Score (Próximo a 1 é melhor)': [f"{met_xgb['R2']:.3f}", f"{met_lstm['R2']:.3f}"]
            })
            st.dataframe(df_metricas, use_container_width=True, hide_index=True)
            
            st.subheader(f"Projeção Futura Comparativa ({dias_predicao} dias)")
            fig_ml = go.Figure()
            dois_anos = df[df['Date'] >= (df['Date'].max() - timedelta(days=730))]
            
            fig_ml.add_trace(go.Scatter(x=dois_anos['Date'], y=dois_anos['Close'], line=dict(color='gray', width=1.5), name='Histórico Real'))
            fig_ml.add_trace(go.Scatter(x=df_xgb['Date'], y=df_xgb['Predicao'], line=dict(color='red', width=2, dash='dash'), name='Predição XGBoost'))
            fig_ml.add_trace(go.Scatter(x=df_lstm['Date'], y=df_lstm['Predicao'], line=dict(color='blue', width=2, dash='dot'), name='Predição LSTM'))
            
            fig_ml.update_layout(
                height=500, 
                template="plotly_white", 
                margin=dict(l=0, r=0, t=40, b=0), 
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
            )
            st.plotly_chart(fig_ml, use_container_width=True)
            
            # --- BOTÃO DE EXPORTAÇÃO CSV ---
            st.markdown("---")
            st.subheader("📥 Exportar Dados")
            
            # Preparar o dataframe unificado para exportação
            df_hist_export = df[['Date', 'Close']].rename(columns={'Close': 'Preco_Real'})
            df_xgb_export = df_xgb.rename(columns={'Predicao': 'Predicao_XGBoost'})
            df_lstm_export = df_lstm.rename(columns={'Predicao': 'Predicao_LSTM'})
            
            # Unir os dados pela data
            df_export = pd.merge(df_hist_export, df_xgb_export, on='Date', how='outer')
            df_export = pd.merge(df_export, df_lstm_export, on='Date', how='outer')
            df_export = df_export.sort_values('Date').reset_index(drop=True)
            
            # Converter para CSV
            csv = df_export.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="Descarregar Dados e Predições (CSV)",
                data=csv,
                file_name=f"projecao_{ticker_symbol}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )

        # ==========================================
        # ABA 3: AGENTE FINANCEIRO
        # ==========================================
        with aba_ia:
            st.subheader("Relatório Executivo: IA e Quantitativo")
            if not api_key:
                st.warning("Configure a Chave da API do Google AI Studio na barra lateral.")
            else:
                with st.spinner("Sintetizando dados, coletando notícias e formulando o relatório..."):
                    try:
                        genai.configure(api_key=api_key)
                        agente = genai.GenerativeModel(model_name='gemini-2.5-pro')
                        
                        preco_atual = df['Close'].iloc[-1]
                        variacao = ((preco_atual - df['Close'].iloc[-90]) / df['Close'].iloc[-90]) * 100 if len(df) > 90 else 0
                        volatilidade = df['Close'].tail(30).std()
                        
                        # --- SOLUÇÃO ROBUSTA PARA NOTÍCIAS (FALLBACK) ---
                        noticias_brutas = yf.Ticker(ticker_symbol).news
                        
                        # Se não encontrar notícias (comum em contratos futuros), busca em ETFs correlacionados
                        if not noticias_brutas:
                            if ticker_symbol == 'HG=F': noticias_brutas = yf.Ticker('CPER').news # ETF Cobre
                            elif ticker_symbol == 'GC=F': noticias_brutas = yf.Ticker('GLD').news # ETF Ouro
                            elif ticker_symbol == 'SI=F': noticias_brutas = yf.Ticker('SLV').news # ETF Prata
                            elif ticker_symbol in ['BZ=F', 'CL=F']: noticias_brutas = yf.Ticker('USO').news # ETF Petróleo
                        
                        if noticias_brutas:
                            manchetes = "\n".join([f"- {n.get('title', 'Sem título')} (Fonte: {n.get('publisher', 'Desconhecida')})" for n in noticias_brutas[:5]])
                        else:
                            manchetes = "Notícias específicas não encontradas. Por favor, baseie a análise no cenário macroeconômico atual do setor."

                        prompt = f"""
                        Atue como um Analista Quantitativo Sênior e Estrategista Macroeconômico de um fundo de investimentos tier-1.
                        Sua tarefa é redigir um relatório analítico executivo sobre o ativo {ticker_symbol}.
                        
                        REGRAS CRÍTICAS DE FORMATAÇÃO:
                        1. NÃO escreva nenhuma frase introdutória ou saudações (como "Aqui está o relatório" ou "Com certeza"). Comece o texto DIRETAMENTE com o título "### 1. Cenário Macroeconômico e Geopolítico Atual".
                        2. NÃO utilize formatação LaTeX matemática que possa quebrar o código (não use \c ou \~).
                        3. Para valores monetários, use a sigla USD antes do número (ex: USD 4.50) para evitar bugs de renderização com múltiplos símbolos de cifrão.
                        
                        DADOS DE MERCADO ATUAIS:
                        - Preço Atual: USD {preco_atual:.2f}
                        - Variação (3 Meses): {variacao:.2f}%
                        - Volatilidade (Desvio Padrão 30d): USD {volatilidade:.2f}
                        
                        MANCHETES E EVENTOS DESTA SEMANA (Setor/Ativo):
                        {manchetes}
                        
                        DADOS PREDITIVOS (Projeção para {dias_predicao} dias):
                        - XGBoost: USD {df_xgb['Predicao'].iloc[-1]:.2f} | MAE: {met_xgb['MAE']:.2f}, R²: {met_xgb['R2']:.2f}
                        - LSTM: USD {df_lstm['Predicao'].iloc[-1]:.2f} | MAE: {met_lstm['MAE']:.2f}, R²: {met_lstm['R2']:.2f}
                        *(Nota: Valores de R² baixos ou negativos indicam que os modelos estão tendo dificuldade de capturar a tendência e as notícias fundamentais devem sobrepor a análise puramente técnica).*
                        
                        ESTRUTURA OBRIGATÓRIA:
                        ### 1. Cenário Macroeconômico e Geopolítico Atual
                        ### 2. Principais Impulsionadores de Preço (Drivers)
                        ### 3. Impacto das Notícias e Eventos da Semana (Analise as manchetes fornecidas)
                        ### 4. Avaliação dos Modelos Quantitativos (Analise as falhas ou acertos baseados no R²)
                        ### 5. Perspectivas e Riscos Futuros
                        ### 6. Conclusão Executiva
                        """
                        
                        resposta = agente.generate_content(prompt)
                        st.write(resposta.text)
                        
                        st.markdown("---")
                        st.caption("⚠️ **Aviso Legal:** Este relatório é gerado por Inteligência Artificial a partir de modelos estatísticos. Não constitui recomendação de investimento.")
                        
                    except Exception as e:
                        st.error(f"Erro na comunicação com a API: {e}")