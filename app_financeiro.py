# --- IMPORTAÇÕES ---
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import plotly.graph_objects as go
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler # Mudamos para Standard (ideal para retornos percentuais)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import google.generativeai as genai

# ==========================================
# CONFIGURAÇÃO INICIAL DA PÁGINA
# ==========================================
st.set_page_config(page_title="Quant & IA Direcional", layout="wide")
st.title("📈 Análise Institucional: Probabilidade Multiclasse e IA")

# ==========================================
# BARRA LATERAL (MENU DE INPUTS)
# ==========================================
st.sidebar.header("Parâmetros da Análise")
ticker_symbol = st.sidebar.text_input("Ticker (ex: VALE3.SA, IAU, AAPL):", "IAU").upper()
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
# FUNÇÃO 2: XGBOOST (MULTICLASSE E ESTACIONÁRIO)
# ==========================================
@st.cache_resource(show_spinner=False) 
def prever_xgboost_class(df, dias_futuros):
    df_ml = df[['Date', 'Close']].copy()
    
    # ESTACIONARIEDADE: Trabalhando com retornos e proporções em vez de preços absolutos
    df_ml['Retorno_Diario'] = df_ml['Close'].pct_change()
    
    lags = [1, 5, 10, 21, 60] 
    for lag in lags:
        df_ml[f'Lag_Ret_{lag}'] = df_ml['Retorno_Diario'].shift(lag)
        
    # Distância Percentual das Médias Móveis (Isso é estacionário)
    df_ml['Dist_SMA_20'] = (df_ml['Close'] - df_ml['Close'].rolling(20).mean()) / df_ml['Close'].rolling(20).mean()
    df_ml['Dist_SMA_50'] = (df_ml['Close'] - df_ml['Close'].rolling(50).mean()) / df_ml['Close'].rolling(50).mean()
    df_ml['Dist_SMA_200'] = (df_ml['Close'] - df_ml['Close'].rolling(200).mean()) / df_ml['Close'].rolling(200).mean()
    
    df_ml['Volatilidade_21'] = df_ml['Retorno_Diario'].rolling(21).std()
    
    delta = df_ml['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_ml['RSI_14'] = 100 - (100 / (1 + rs))
    
    # REDUÇÃO DE RUÍDO: Target Multiclasse Dinâmico (0 = Baixa, 1 = Neutro, 2 = Alta)
    retorno_futuro = (df_ml['Close'].shift(-dias_futuros) - df_ml['Close']) / df_ml['Close']
    
    # Calcula a volatilidade esperada para o período. Consideramos "Ruído/Neutro" o que ficar dentro de 0.5 Desvios Padrões
    vol_periodo = df_ml['Retorno_Diario'].std() * np.sqrt(dias_futuros)
    limite_ruido = vol_periodo * 0.5 
    
    condicoes = [
        (retorno_futuro > limite_ruido),   # Subiu além do ruído = 2 (Alta)
        (retorno_futuro < -limite_ruido)   # Caiu além do ruído = 0 (Baixa)
    ]
    escolhas = [2, 0]
    df_ml['Target'] = np.select(condicoes, escolhas, default=1) # O que sobrar é 1 (Neutro)
    
    df_ml = df_ml.dropna() 
    features = [f'Lag_Ret_{lag}' for lag in lags] + ['Dist_SMA_20', 'Dist_SMA_50', 'Dist_SMA_200', 'Volatilidade_21', 'RSI_14']
    
    df_treino = df_ml.iloc[:-dias_futuros]
    X = df_treino[features] 
    y = df_treino['Target'] 
    
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    # CONFIGURAÇÃO MULTICLASSE
    parametros = {'objective': 'multi:softprob', 'num_class': 3, 'n_estimators': 150, 'learning_rate': 0.05, 'max_depth': 4, 'eval_metric': 'mlogloss'}
    
    modelo_aval = xgb.XGBClassifier(**parametros)
    modelo_aval.fit(X_train, y_train)
    pred_teste = modelo_aval.predict(X_test)
    
    # Acurácia simples. (Num problema de 3 classes, aleatório é 33%, então valores > 45% já são muito bons)
    acuracia = accuracy_score(y_test, pred_teste)
    
    modelo_final = xgb.XGBClassifier(**parametros)
    modelo_final.fit(X, y)
    
    ultimo_dado = df_ml.iloc[-1:][features]
    probabilidades = modelo_final.predict_proba(ultimo_dado)[0] * 100 # Array: [Prob Baixa, Prob Neutra, Prob Alta]
    
    return probabilidades, acuracia

# ==========================================
# FUNÇÃO 3: LSTM (MULTICLASSE E ESTACIONÁRIA)
# ==========================================
@st.cache_resource(show_spinner=False)
def prever_lstm_class(df, dias_futuros):
    df_ml = df[['Date', 'Close']].copy()
    
    df_ml['Retorno_Diario'] = df_ml['Close'].pct_change()
    
    # Mesmo cálculo multiclasse de redução de ruído
    retorno_futuro = (df_ml['Close'].shift(-dias_futuros) - df_ml['Close']) / df_ml['Close']
    vol_periodo = df_ml['Retorno_Diario'].std() * np.sqrt(dias_futuros)
    limite_ruido = vol_periodo * 0.5 
    
    condicoes = [(retorno_futuro > limite_ruido), (retorno_futuro < -limite_ruido)]
    df_ml['Target'] = np.select(condicoes, [2, 0], default=1)
    
    df_ml = df_ml.dropna()
    
    # Usamos o StandardScaler porque retornos orbitam em torno de 0, não entre 0 e 1.
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_ml['Retorno_Diario'].values.reshape(-1, 1))
    
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
        # Dense 3 e Softmax para retornar as 3 probabilidades
        modelo.add(Dense(units=3, activation='softmax')) 
        # sparse_categorical_crossentropy é a função de perda correta para alvos 0, 1, 2
        modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return modelo
    
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    modelo_aval = construir_modelo()
    modelo_aval.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=16, epochs=50, verbose=0, callbacks=[early_stop])
    
    pred_teste_prob = modelo_aval.predict(X_test, verbose=0)
    pred_teste_classe = np.argmax(pred_teste_prob, axis=1) # Pega o índice com maior probabilidade
    
    acuracia = accuracy_score(y_test, pred_teste_classe)
    
    modelo_final = construir_modelo()
    modelo_final.fit(X_treino, y_treino, batch_size=16, epochs=50, verbose=0) 
    
    ultimos_dias = scaled_data[-lookback:]
    x_pred = np.reshape(ultimos_dias, (1, lookback, 1))
    probabilidades = modelo_final.predict(x_pred, verbose=0)[0] * 100 
    
    return probabilidades, acuracia

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

        aba_eda, aba_ml, aba_ia = st.tabs(["📊 Análise Exploratória", "🤖 Machine Learning Multiclasse", "🧠 Agente Financeiro"])

        # ==========================================
        # ABA 1: ANÁLISE EXPLORATÓRIA (ESTILIZADA E PROFISSIONAL)
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
            fig_candle.update_layout(xaxis_rangeslider_visible=False, height=400, template="plotly_white", margin=dict(l=0, r=0, t=30, b=0), xaxis_title="Data", yaxis_title=f"Preço ({moeda})")
            st.plotly_chart(fig_candle, use_container_width=True) 

            col_dist, col_box = st.columns(2)
            with col_dist:
                st.subheader("Distribuição de Frequência")
                fig_dist = go.Figure()
                fig_dist.add_trace(go.Histogram(x=df['Close'], marker_color='lightblue', opacity=0.7))
                media_val, mediana_val = df['Close'].mean(), df['Close'].median()
                
                # Restaurando as linhas verticais com anotações de valores
                fig_dist.add_vline(x=media_val, line_dash="dash", line_color="red", 
                                  annotation_text=f"Média: {moeda} {media_val:.2f}", annotation_position="top right")
                fig_dist.add_vline(x=mediana_val, line_dash="dash", line_color="green", 
                                  annotation_text=f"Mediana: {moeda} {mediana_val:.2f}", annotation_position="bottom right")
                
                fig_dist.update_layout(height=350, template="plotly_white", margin=dict(l=0, r=0, t=30, b=0), showlegend=False, xaxis_title=f"Preço ({moeda})", yaxis_title="Dias")
                st.plotly_chart(fig_dist, use_container_width=True)

            with col_box:
                st.subheader("Box Plot de Preços")
                fig_box = go.Figure()
                fig_box.add_trace(go.Box(
                    y=df['Close'], 
                    name=ticker_symbol, 
                    marker_color='tan', 
                    boxpoints='outliers', 
                    yhoverformat=",.2f",
                    boxmean=True # <-- Linha pontilhada da Média
                ))
                fig_box.update_layout(height=350, template="plotly_white", margin=dict(l=0, r=0, t=30, b=0), yaxis_title=f"Preço ({moeda})")
                fig_box.update_yaxes(tickprefix=f"{moeda} ", tickformat=".2f") 
                st.plotly_chart(fig_box, use_container_width=True)
            
            # --- RESTAURANDO A TABELA DE COMPARAÇÃO ANUAL COM CORES ---
            st.markdown("---")
            st.subheader("Comparação Anual de Preços e Variação (%)")
            
            # 1. Cria o DataFrame resumido (groupby por ano e média)
            df_yoy = df.groupby('Year')['Close'].mean().reset_index()
            col_preco_medio = f'Preço Médio ({moeda})'
            df_yoy.columns = ['Ano', col_preco_medio]
            # pct_change calcula a variação percentual
            df_yoy['Variação (%)'] = df_yoy[col_preco_medio].pct_change() * 100 
            
            # 2. Cria o DataFrame formatado para exibição (Moeda e símbolo %)
            df_yoy_formatado = df_yoy.copy()
            # Formatação profissional com a moeda dinâmica e 2 casas decimais
            df_yoy_formatado[col_preco_medio] = df_yoy_formatado[col_preco_medio].apply(lambda x: f"{moeda} {x:.2f}")
            
            # Função auxiliar para formatação percentual, cuidando de valores NaN
            def formatar_percentual(x):
                if pd.notnull(x): return f"{x:.2f}%"
                return "-"
            df_yoy_formatado['Variação (%)'] = df_yoy_formatado['Variação (%)'].apply(formatar_percentual)
            
            # --- NOVO: Estilização Condicional (Pandas Styler) ---
            # Função para definir a cor de fundo (background) baseada no valor
            def estilizar_variacao(val):
                color = '#ffcccb' if val < 0 else '#d4edda' # Vermelho claro se < 0, Verde claro se >= 0
                return f'background-color: {color}'
            
            # Função para definir a cor do texto baseada no valor
            def estilizar_texto(val):
                color = 'red' if val < 0 else 'green' # Texto Vermelho se < 0, Verde se >= 0
                return f'color: {color}'
            
            # Aplica as funções de estilização na coluna 'Variação (%)' do DataFrame
            df_estilizado = df_yoy_formatado.style.applymap(estilizar_variacao, subset=['Variação (%)'])\
                                                 .applymap(estilizar_texto, subset=['Variação (%)'])
            
            # Exibe o DataFrame estilizado
            st.dataframe(df_estilizado, use_container_width=True, hide_index=True)

        # ==========================================
        # ABA 2: MACHINE LEARNING MULTICLASSE
        # ==========================================
        with aba_ml:
            st.subheader("Análise de Tendência e Contexto Macroeconômico")
            dois_anos_atras = df['Date'].max() - timedelta(days=730)
            df_sma = df[df['Date'] >= dois_anos_atras].copy()
            df_sma['SMA_20'] = df_sma['Close'].rolling(window=20).mean()
            df_sma['SMA_50'] = df_sma['Close'].rolling(window=50).mean()
            df_sma['SMA_200'] = df_sma['Close'].rolling(window=200).mean()
            
            fig_sma = go.Figure()
            fig_sma.add_trace(go.Scatter(x=df_sma['Date'], y=df_sma['Close'], mode='lines', name='Preço Real', line=dict(color='gray', width=1.5)))
            fig_sma.add_trace(go.Scatter(x=df_sma['Date'], y=df_sma['SMA_20'], mode='lines', name='SMA 20', line=dict(color='blue', width=1)))
            fig_sma.add_trace(go.Scatter(x=df_sma['Date'], y=df_sma['SMA_50'], mode='lines', name='SMA 50', line=dict(color='orange', width=1)))
            fig_sma.add_trace(go.Scatter(x=df_sma['Date'], y=df_sma['SMA_200'], mode='lines', name='SMA 200', line=dict(color='red', width=2, dash='dash')))
            fig_sma.update_layout(height=450, template="plotly_white", margin=dict(l=0, r=0, t=30, b=0), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
            st.plotly_chart(fig_sma, use_container_width=True)
            
            st.markdown("---")
            with st.spinner("Computando modelos estacionários Multiclasse... (Aguarde)"):
                prob_xgb, acc_xgb = prever_xgboost_class(df, dias_predicao)
                prob_lstm, acc_lstm = prever_lstm_class(df, dias_predicao)
            
            st.subheader(f"🎯 Probabilidade Direcional Multiclasse (Próximos {dias_predicao} dias)")
            st.write("Em modelos de 3 classes (Alta, Neutro, Baixa), a linha base de acerto aleatório é 33%.")
            
            # Função auxiliar para criar barras de probabilidade empilhadas
            def criar_barra_probabilidade(probs, titulo):
                fig = go.Figure(go.Bar(
                    y=['Probabilidade'], x=[probs[0]], name='Baixa', orientation='h', marker=dict(color='#ffcccb')
                ))
                fig.add_trace(go.Bar(
                    y=['Probabilidade'], x=[probs[1]], name='Neutro / Lateral', orientation='h', marker=dict(color='#f0f0f0')
                ))
                fig.add_trace(go.Bar(
                    y=['Probabilidade'], x=[probs[2]], name='Alta', orientation='h', marker=dict(color='#d4edda')
                ))
                fig.update_layout(barmode='stack', title=titulo, height=200, margin=dict(l=0, r=0, t=40, b=0), 
                                  xaxis=dict(range=[0, 100], ticksuffix="%"))
                return fig

            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(criar_barra_probabilidade(prob_xgb, f"XGBoost Classifier (Acc: {acc_xgb*100:.1f}%)"), use_container_width=True)
            with col2:
                st.plotly_chart(criar_barra_probabilidade(prob_lstm, f"LSTM Deep Learning (Acc: {acc_lstm*100:.1f}%)"), use_container_width=True)

        # ==========================================
        # ABA 3: AGENTE FINANCEIRO
        # ==========================================
        with aba_ia:
            st.subheader("Relatório Executivo: IA e Quantitativo")
            if not api_key:
                st.warning("Configure a Chave da API do Google AI Studio na barra lateral.")
            else:
                with st.spinner("Sintetizando probabilidades multiclasse e formulando relatório..."):
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
                            elif ticker_symbol == 'IAU': noticias_brutas = yf.Ticker('GLD').news 
                        
                        if noticias_brutas:
                            manchetes = "\n".join([f"- {n.get('title', 'Sem título')} (Fonte: {n.get('publisher', 'Desconhecida')})" for n in noticias_brutas[:5]])
                        else:
                            manchetes = "Notícias não encontradas."

                        prompt = f"""
                        Atue como um Analista Quantitativo Sênior e Estrategista Macroeconômico de um fundo de investimentos tier-1.
                        Sua tarefa é redigir um relatório analítico executivo sobre {ticker_symbol}.
                        
                        REGRAS: Comece DIRETAMENTE no texto principal. Sem saudações. Sem LaTeX matemático. Use a sigla {moeda}.
                        
                        DADOS ATUAIS:
                        - Preço Atual: {moeda} {preco_atual:.2f}
                        - Variação (3 Meses): {variacao:.2f}%
                        - Volatilidade 30d: {moeda} {volatilidade:.2f}
                        
                        MANCHETES:
                        {manchetes}
                        
                        DADOS PREDITIVOS MULTICLASSE ({dias_predicao} dias):
                        Nota: Como é um problema de 3 direções, a linha base de acerto aleatório é 33%. Acurácias acima de 45% já mostram extração real de sinal (Edge quantitativo).
                        
                        - XGBoost (Acurácia Histórica: {acc_xgb*100:.1f}%):
                          Chance de ALTA: {prob_xgb[2]:.1f}% | NEUTRO/LATERAL: {prob_xgb[1]:.1f}% | BAIXA: {prob_xgb[0]:.1f}%
                          
                        - LSTM (Acurácia Histórica: {acc_lstm*100:.1f}%):
                          Chance de ALTA: {prob_lstm[2]:.1f}% | NEUTRO/LATERAL: {prob_lstm[1]:.1f}% | BAIXA: {prob_lstm[0]:.1f}%
                        
                        ESTRUTURA:
                        ### 1. Cenário Macroeconômico e Geopolítico Atual
                        ### 2. Principais Impulsionadores (Drivers)
                        ### 3. Impacto das Notícias
                        ### 4. Avaliação Multiclasse dos Modelos (Interprete se os modelos estão concordando com a tendência ou prevendo correção/lateralização)
                        ### 5. Conclusão Executiva
                        """
                        
                        resposta = agente.generate_content(prompt)
                        st.write(resposta.text)
                        
                    except Exception as e:
                        st.error(f"Erro na API: {e}")