# --- IMPORTAÇÕES ---
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import plotly.graph_objects as go
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import google.generativeai as genai

# ==========================================
# CONFIGURAÇÃO INICIAL DA PÁGINA
# ==========================================
st.set_page_config(page_title="Análise Quantitativa & IA", layout="wide")
st.title("📈 Análise de Ativos, Probabilidade Direcional e IA")

# ==========================================
# BARRA LATERAL (MENU DE INPUTS)
# ==========================================
st.sidebar.header("Parâmetros da Análise")
ticker_symbol = st.sidebar.text_input("Ticker (ex: VALE3.SA, IAU, AAPL):", "VALE3.SA").upper()
anos_historico = st.sidebar.number_input("Anos de histórico:", min_value=1, max_value=20, value=10)
dias_predicao = st.sidebar.slider("Janela de Predição (Dias úteis):", 10, 90, 30)

st.sidebar.markdown("---")
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
else:
    api_key = st.sidebar.text_input("Chave API (Google AI Studio):", type="password")
st.sidebar.markdown("---")

# ==========================================
# FUNÇÃO 1: COLETA E PRÉ-PROCESSAMENTO
# ==========================================
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

# ==========================================
# FUNÇÃO 2: MODELO XGBOOST (CLASSIFICAÇÃO COM MACRO TRENDS)
# ==========================================
@st.cache_resource(show_spinner=False) 
def prever_xgboost_class(df, dias_futuros):
    df_ml = df[['Date', 'Close']].copy()
    
    # 1. Lags de curto e longo prazo (até 1 trimestre)
    lags = [1, 5, 10, 21, 60] 
    for lag in lags:
        df_ml[f'Lag_{lag}'] = df_ml['Close'].shift(lag)
        
    # 2. Médias Móveis de Curto, Médio e Longo Prazo
    df_ml['SMA_20'] = df_ml['Close'].rolling(window=20).mean()
    df_ml['SMA_50'] = df_ml['Close'].rolling(window=50).mean()   # Tendência de Médio Prazo
    df_ml['SMA_200'] = df_ml['Close'].rolling(window=200).mean() # Tendência Macroeconômica Primária
    
    # 3. Distância do preço atual para a SMA 200 (Mede se o ativo está muito "esticado")
    df_ml['Dist_SMA_200'] = (df_ml['Close'] - df_ml['SMA_200']) / df_ml['SMA_200']
    
    # 4. Volatilidade (Desvio padrão dos retornos diários no último mês)
    df_ml['Volatilidade_21'] = df_ml['Close'].pct_change().rolling(window=21).std()
    
    # 5. RSI Clássico (Força Relativa)
    delta = df_ml['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_ml['RSI_14'] = 100 - (100 / (1 + rs))
    
    # TARGET: 1 se o preço no futuro for maior que hoje, senão 0
    df_ml['Target'] = (df_ml['Close'].shift(-dias_futuros) > df_ml['Close']).astype(int)
    
    # Remove as primeiras 200 linhas (que ficam vazias por causa da SMA_200) e os feriados sem dados
    df_ml = df_ml.dropna() 
    
    features = [f'Lag_{lag}' for lag in lags] + ['SMA_20', 'SMA_50', 'SMA_200', 'Dist_SMA_200', 'Volatilidade_21', 'RSI_14']
    
    df_treino = df_ml.iloc[:-dias_futuros]
    
    X = df_treino[features] 
    y = df_treino['Target'] 
    
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    parametros = {'objective': 'binary:logistic', 'n_estimators': 150, 'learning_rate': 0.05, 'max_depth': 4, 'eval_metric': 'logloss'}
    
    modelo_aval = xgb.XGBClassifier(**parametros)
    modelo_aval.fit(X_train, y_train)
    pred_teste = modelo_aval.predict(X_test)
    
    metricas = {
        'Acurácia': accuracy_score(y_test, pred_teste),
        'Precisão (Acerto de Altas)': precision_score(y_test, pred_teste, zero_division=0)
    }
    
    modelo_final = xgb.XGBClassifier(**parametros)
    modelo_final.fit(X, y)
    
    ultimo_dado = df_ml.iloc[-1:][features]
    prob_alta = modelo_final.predict_proba(ultimo_dado)[0][1] * 100 
    
    return prob_alta, metricas

# ==========================================
# FUNÇÃO 3: REDE NEURAL LSTM (CLASSIFICAÇÃO TRIMESTRAL)
# ==========================================
@st.cache_resource(show_spinner=False)
def prever_lstm_class(df, dias_futuros):
    df_ml = df[['Date', 'Close']].copy()
    
    df_ml['Target'] = (df_ml['Close'].shift(-dias_futuros) > df_ml['Close']).astype(int)
    df_ml = df_ml.dropna()
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df_ml['Close'].values.reshape(-1, 1))
    
    # Aumentamos a janela de memória para 60 dias (1 trimestre) para capturar ciclos mais longos
    lookback = 60 
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0]) 
        y.append(df_ml['Target'].iloc[i])      
        
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    X_treino = X[:-dias_futuros]
    y_treino = y[:-dias_futuros]
    
    split_idx = int(len(X_treino) * 0.8)
    X_train, y_train = X_treino[:split_idx], y_treino[:split_idx]
    X_test, y_test = X_treino[split_idx:], y_treino[split_idx:]
    
    def construir_modelo():
        modelo = Sequential()
        modelo.add(LSTM(units=32, return_sequences=False, input_shape=(lookback, 1)))
        modelo.add(Dropout(0.2))
        modelo.add(Dense(units=1, activation='sigmoid')) 
        modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        return modelo
    
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    modelo_aval = construir_modelo()
    modelo_aval.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=16, epochs=50, verbose=0, callbacks=[early_stop])
    
    pred_teste_prob = modelo_aval.predict(X_test, verbose=0).flatten()
    pred_teste_classe = (pred_teste_prob > 0.5).astype(int)
    
    metricas = {
        'Acurácia': accuracy_score(y_test, pred_teste_classe),
        'Precisão (Acerto de Altas)': precision_score(y_test, pred_teste_classe, zero_division=0)
    }
    
    modelo_final = construir_modelo()
    modelo_final.fit(X_treino, y_treino, batch_size=16, epochs=50, verbose=0) 
    
    ultimos_dias = scaled_data[-lookback:]
    x_pred = np.reshape(ultimos_dias, (1, lookback, 1))
    prob_alta = modelo_final.predict(x_pred, verbose=0)[0][0] * 100 
    
    return prob_alta, metricas

# ==========================================
# MOTOR PRINCIPAL (FRONT-END)
# ==========================================
if st.sidebar.button("Analisar Ativo"):
    with st.spinner("Baixando dados do Yahoo Finance..."):
        df = carregar_e_processar_dados(ticker_symbol, anos_historico)
        
    if df.empty:
        st.error("Nenhum dado encontrado. Verifique o Ticker ou sua conexão.")
    else:
        try:
            moeda = yf.Ticker(ticker_symbol).fast_info.currency
        except:
            moeda = "BRL" if ticker_symbol.endswith(".SA") else "USD"

        aba_eda, aba_ml, aba_ia = st.tabs(["📊 Análise Exploratória (EDA)", "🤖 Machine Learning Direcional", "🧠 Agente Financeiro"])

        # ==========================================
        # ABA 1: ANÁLISE EXPLORATÓRIA
        # ==========================================
        with aba_eda:
            st.subheader(f"Métricas Principais: {ticker_symbol}")
            c1, c2, c3, c4 = st.columns(4) 
            c1.metric("Preço Atual", f"{moeda} {df['Close'].iloc[-1]:.2f}")
            c2.metric("Média Histórica", f"{moeda} {df['Close'].mean():.2f}")
            c3.metric("Máxima", f"{moeda} {df['High'].max():.2f}")
            c4.metric("Mínima", f"{moeda} {df['Low'].min():.2f}")
            
            st.markdown("---")
            st.subheader("Gráfico de Velas (Últimos 6 Meses)")
            seis_meses_atras = df['Date'].max() - timedelta(days=6*30)
            df_6m = df[df['Date'] >= seis_meses_atras]
            fig_candle = go.Figure(data=[go.Candlestick(x=df_6m['Date'], open=df_6m['Open'], high=df_6m['High'], low=df_6m['Low'], close=df_6m['Close'])])
            fig_candle.update_layout(
                xaxis_rangeslider_visible=False, height=400, template="plotly_white", margin=dict(l=0, r=0, t=30, b=0),
                xaxis_title="Data",
                yaxis_title=f"Preço do Ativo ({moeda})"
            )
            st.plotly_chart(fig_candle, use_container_width=True) 

            col_dist, col_box = st.columns(2)
            with col_dist:
                st.subheader("Distribuição de Frequência")
                fig_dist = go.Figure()
                fig_dist.add_trace(go.Histogram(x=df['Close'], marker_color='lightblue', opacity=0.7))
                media, mediana = df['Close'].mean(), df['Close'].median()
                fig_dist.add_vline(x=media, line_dash="dash", line_color="red", annotation_text=f"Média: {media:.2f}", annotation_position="top right")
                fig_dist.add_vline(x=mediana, line_dash="dash", line_color="green", annotation_text=f"Mediana: {mediana:.2f}", annotation_position="bottom right")
                fig_dist.update_layout(
                    height=350, template="plotly_white", margin=dict(l=0, r=0, t=30, b=0), showlegend=False,
                    xaxis_title=f"Preço de Fechamento ({moeda})",
                    yaxis_title="Número de Dias"
                )
                st.plotly_chart(fig_dist, use_container_width=True)

            with col_box:
                st.subheader("Box Plot de Preços")
                fig_box = go.Figure()
                fig_box.add_trace(go.Box(
                    y=df['Close'], 
                    name=ticker_symbol, 
                    marker_color='tan',
                    boxpoints='outliers', 
                    yhoverformat=",.2f"   
                ))
                fig_box.update_layout(
                    height=350, template="plotly_white", margin=dict(l=0, r=0, t=30, b=0),
                    yaxis_title=f"Preço de Fechamento ({moeda})"
                )
                fig_box.update_yaxes(tickprefix=f"{moeda} ", tickformat=".2f") 
                st.plotly_chart(fig_box, use_container_width=True)
            
            st.markdown("---")
            st.subheader("Comparação Anual de Preços e Variação (%)")
            df_yoy = df.groupby('Year')['Close'].mean().reset_index()
            df_yoy.columns = ['Ano', f'Preço Médio ({moeda})']
            df_yoy['Variação (%)'] = df_yoy[f'Preço Médio ({moeda})'].pct_change() * 100 
            
            df_yoy_formatado = df_yoy.copy()
            df_yoy_formatado[f'Preço Médio ({moeda})'] = df_yoy_formatado[f'Preço Médio ({moeda})'].apply(lambda x: f"{moeda} {x:.2f}")
            df_yoy_formatado['Variação (%)'] = df_yoy_formatado['Variação (%)'].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "-")
            st.dataframe(df_yoy_formatado, use_container_width=True, hide_index=True)

        # ==========================================
        # ABA 2: MACHINE LEARNING (CLASSIFICAÇÃO) E MÉDIAS MÓVEIS
        # ==========================================
        with aba_ml:
            # NOVO GRÁFICO: Contexto de Tendência Macro (SMA)
            st.subheader("Análise de Tendência e Contexto Macroeconômico")
            st.write("Visualização do preço real cruzado com as principais médias móveis (Curto, Médio e Longo prazo).")
            
            # Filtra os últimos 2 anos para o gráfico não ficar esmagado
            dois_anos_atras = df['Date'].max() - timedelta(days=730)
            df_sma = df[df['Date'] >= dois_anos_atras].copy()
            df_sma['SMA_20'] = df_sma['Close'].rolling(window=20).mean()
            df_sma['SMA_50'] = df_sma['Close'].rolling(window=50).mean()
            df_sma['SMA_200'] = df_sma['Close'].rolling(window=200).mean()
            
            fig_sma = go.Figure()
            fig_sma.add_trace(go.Scatter(x=df_sma['Date'], y=df_sma['Close'], mode='lines', name='Preço Real', line=dict(color='gray', width=1.5)))
            fig_sma.add_trace(go.Scatter(x=df_sma['Date'], y=df_sma['SMA_20'], mode='lines', name='SMA 20 (Curto)', line=dict(color='blue', width=1)))
            fig_sma.add_trace(go.Scatter(x=df_sma['Date'], y=df_sma['SMA_50'], mode='lines', name='SMA 50 (Médio)', line=dict(color='orange', width=1)))
            fig_sma.add_trace(go.Scatter(x=df_sma['Date'], y=df_sma['SMA_200'], mode='lines', name='SMA 200 (Longo)', line=dict(color='red', width=2, dash='dash')))
            
            fig_sma.update_layout(
                height=450, template="plotly_white", margin=dict(l=0, r=0, t=30, b=0),
                xaxis_title="Data", yaxis_title=f"Preço ({moeda})",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
            )
            st.plotly_chart(fig_sma, use_container_width=True)
            
            st.markdown("---")

            with st.spinner("Analisando probabilidades com XGBoost e LSTM... (Aguarde)"):
                prob_xgb, met_xgb = prever_xgboost_class(df, dias_predicao)
                prob_lstm, met_lstm = prever_lstm_class(df, dias_predicao)
            
            st.subheader(f"🎯 Probabilidade de Fechar em Alta (Daqui a {dias_predicao} dias)")
            
            col_gauge1, col_gauge2 = st.columns(2)
            
            with col_gauge1:
                fig_xgb = go.Figure(go.Indicator(
                    mode = "gauge+number", value = prob_xgb, title = {'text': "XGBoost Classifier (%)"},
                    gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "darkred"},
                             'steps': [
                                 {'range': [0, 45], 'color': "#ffcccb"},   
                                 {'range': [45, 55], 'color': "#f0f0f0"},  
                                 {'range': [55, 100], 'color': "#d4edda"}  
                             ]}
                ))
                fig_xgb.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig_xgb, use_container_width=True)
                
            with col_gauge2:
                fig_lstm = go.Figure(go.Indicator(
                    mode = "gauge+number", value = prob_lstm, title = {'text': "LSTM Classifier (%)"},
                    gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "darkblue"},
                             'steps': [
                                 {'range': [0, 45], 'color': "#ffcccb"}, 
                                 {'range': [45, 55], 'color': "#f0f0f0"}, 
                                 {'range': [55, 100], 'color': "#d4edda"}
                             ]}
                ))
                fig_lstm.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig_lstm, use_container_width=True)

            st.markdown("---")
            st.subheader("🏆 Confiabilidade Histórica dos Modelos")
            df_metricas = pd.DataFrame({
                'Modelo': ['XGBoost (Árvores)', 'LSTM (Rede Neural)'],
                'Acurácia Global': [f"{met_xgb['Acurácia']*100:.1f}%", f"{met_lstm['Acurácia']*100:.1f}%"],
                'Precisão (Acerto nas Altas)': [f"{met_xgb['Precisão (Acerto de Altas)']*100:.1f}%", f"{met_lstm['Precisão (Acerto de Altas)']*100:.1f}%"]
            })
            st.dataframe(df_metricas, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            st.subheader("📥 Exportar Histórico de Dados")
            st.write("Faça o download da base de dados histórica utilizada para treinar os modelos.")
            csv = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Descarregar Dados (CSV)",
                data=csv,
                file_name=f"historico_{ticker_symbol}_{datetime.now().strftime('%Y%m%d')}.csv",
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
                with st.spinner("Sintetizando dados, coletando notícias e formulando o relatório direcional..."):
                    try:
                        genai.configure(api_key=api_key)
                        agente = genai.GenerativeModel(model_name='gemini-2.5-pro')
                        
                        preco_atual = df['Close'].iloc[-1]
                        variacao = ((preco_atual - df['Close'].iloc[-90]) / df['Close'].iloc[-90]) * 100 if len(df) > 90 else 0
                        volatilidade = df['Close'].tail(30).std()
                        
                        noticias_brutas = yf.Ticker(ticker_symbol).news
                        if not noticias_brutas:
                            if ticker_symbol == 'HG=F': noticias_brutas = yf.Ticker('CPER').news 
                            elif ticker_symbol == 'GC=F': noticias_brutas = yf.Ticker('GLD').news 
                            elif ticker_symbol == 'SI=F': noticias_brutas = yf.Ticker('SLV').news 
                            elif ticker_symbol in ['BZ=F', 'CL=F']: noticias_brutas = yf.Ticker('USO').news 
                            elif ticker_symbol == 'IAU': noticias_brutas = yf.Ticker('GLD').news # Fallback extra para IAU
                        
                        if noticias_brutas:
                            manchetes = "\n".join([f"- {n.get('title', 'Sem título')} (Fonte: {n.get('publisher', 'Desconhecida')})" for n in noticias_brutas[:5]])
                        else:
                            manchetes = "Notícias específicas não encontradas. Por favor, baseie a análise no cenário macroeconômico atual do setor."

                        prompt = f"""
                        Atue como um Analista Quantitativo Sênior e Estrategista Macroeconômico de um fundo de investimentos tier-1.
                        Sua tarefa é redigir um relatório analítico executivo sobre o ativo {ticker_symbol}.
                        
                        REGRAS CRÍTICAS DE FORMATAÇÃO:
                        1. NÃO escreva nenhuma frase introdutória ou saudações. Comece o texto DIRETAMENTE com o título "### 1. Cenário Macroeconômico e Geopolítico Atual".
                        2. NÃO utilize formatação LaTeX matemática que possa quebrar o código.
                        3. Para valores monetários, use a sigla {moeda} antes do número (ex: {moeda} 4.50).
                        
                        DADOS DE MERCADO ATUAIS:
                        - Preço Atual: {moeda} {preco_atual:.2f}
                        - Variação (3 Meses): {variacao:.2f}%
                        - Volatilidade (Desvio Padrão 30d): {moeda} {volatilidade:.2f}
                        
                        MANCHETES E EVENTOS DESTA SEMANA (Setor/Ativo):
                        {manchetes}
                        
                        DADOS PREDITIVOS (Probabilidade de Fechar em ALTA daqui a {dias_predicao} dias):
                        - XGBoost: {prob_xgb:.1f}% de chance | Histórico de Acurácia: {met_xgb['Acurácia']*100:.1f}%
                        - LSTM: {prob_lstm:.1f}% de chance | Histórico de Acurácia: {met_lstm['Acurácia']*100:.1f}%
                        *(Nota: Valores acima de 55% indicam viés direcional de alta, abaixo de 45% indicam baixa, e entre 45-55% é incerteza/ruído).*
                        
                        ESTRUTURA OBRIGATÓRIA:
                        ### 1. Cenário Macroeconômico e Geopolítico Atual
                        ### 2. Principais Impulsionadores de Preço (Drivers)
                        ### 3. Impacto das Notícias e Eventos da Semana
                        ### 4. Avaliação dos Modelos Quantitativos (Analise as probabilidades direcionais e a acurácia dos modelos)
                        ### 5. Perspectivas e Riscos Futuros
                        ### 6. Conclusão Executiva
                        """
                        
                        resposta = agente.generate_content(prompt)
                        st.write(resposta.text)
                        
                        st.markdown("---")
                        st.caption("⚠️ **Aviso Legal:** Este relatório é gerado por Inteligência Artificial a partir de modelos estatísticos de probabilidade direcional. Não constitui recomendação de investimento.")
                        
                    except Exception as e:
                        st.error(f"Erro na comunicação com a API: {e}")