import os
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import json
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from datetime import datetime, timedelta

# --- CONFIGURAÇÃO DE DIRETÓRIOS DINÂMICOS ---
DIRETORIO_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
RAIZ_PROJETO = os.path.abspath(os.path.join(DIRETORIO_SCRIPTS, ".."))

PASTA_DATA = os.path.join(RAIZ_PROJETO, "data")
PASTA_MODELS = os.path.join(RAIZ_PROJETO, "models")

# Garantir que as pastas existam
os.makedirs(PASTA_DATA, exist_ok=True)
os.makedirs(PASTA_MODELS, exist_ok=True)

# ==============================================================================
# FUNÇÕES DE ML (XGBOOST E LSTM)
# ==============================================================================

def treinar_xgboost_multi(df, dias_futuros):
    """Treina XGBoost com Walk-Forward Validation e retorna (probabilidades, acurácia)."""
    df_ml = df.copy()
    features = [c for c in df_ml.columns if 'Lag' in c or 'Correl' in c or 'Dist' in c or 'Volatilidade' in c]
    
    X = df_ml.dropna().drop(columns=['Target'], errors='ignore')[features]
    y = df_ml.dropna()['Target'].astype(int)
    
    tscv = TimeSeriesSplit(n_splits=5)
    scores = []
    model = XGBClassifier(n_estimators=100, objective='multi:softprob', num_class=3)
    
    for train_idx, test_idx in tscv.split(X):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[test_idx])
        scores.append(accuracy_score(y.iloc[test_idx], preds))
    
    model.fit(X, y) # Treino final
    prob_hoje = model.predict_proba(X.iloc[-1:])[0] * 100
    return model, prob_hoje, np.mean(scores)

def treinar_lstm_multi(df, dias_futuros, scaler):
    """Treina LSTM e retorna (modelo, probabilidades, acurácia)."""
    f_lstm = ['Ret_Ativo', 'Ret_Macro', 'Volatilidade']
    scaled = scaler.transform(df[f_lstm])
    
    lookback = 60
    X, y = [], []
    for i in range(lookback, len(scaled) - dias_futuros):
        X.append(scaled[i-lookback:i])
        y.append(df['Target'].iloc[i])
    
    X, y = np.array(X), np.array(y)
    split = int(len(X) * 0.8)
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(lookback, 3)),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X[:split], y[:split], epochs=5, verbose=0)
    
    loss, acc = model.evaluate(X[split:], y[split:], verbose=0)
    prob_hoje = model.predict(np.reshape(scaled[-lookback:], (1, lookback, 3)), verbose=0)[0] * 100
    return model, prob_hoje, acc

# ==============================================================================
# PIPELINE PRINCIPAL (EXECUTADO PELA DAG)
# ==============================================================================

def executar_pipeline_completo(ticker="QQQ", macro="^TNX"):
    print(f"🚀 Iniciando Pipeline para {ticker}...")
    
    # 1. Ingestão
    end_date = datetime.now()
    start_date = end_date - timedelta(days=10*365)
    data = yf.download([ticker, macro], start=start_date.strftime('%Y-%m-%d'), progress=False)
    if 'Close' in data.columns.levels[0]: data = data['Close']
    data = data.rename(columns={ticker: 'Ativo', macro: 'Macro'}).dropna().ffill()
    
    # 2. Engenharia de Features
    df = data.copy()
    df['Ret_Ativo'] = df['Ativo'].pct_change()
    df['Ret_Macro'] = df['Macro'].pct_change()
    df['Volatilidade'] = df['Ret_Ativo'].rolling(21).std()
    for lag in [1, 5, 21]:
        df[f'Lag_Ativo_{lag}'] = df['Ret_Ativo'].shift(lag)
        df[f'Lag_Macro_{lag}'] = df['Ret_Macro'].shift(lag)
    
    # 3. Target
    ret_futuro = (df['Ativo'].shift(-30) - df['Ativo']) / df['Ativo']
    limite = df['Ret_Ativo'].std() * np.sqrt(30) * 0.5
    df['Target'] = np.select([(ret_futuro > limite), (ret_futuro < -limite)], [2, 0], default=1)
    df.loc[ret_futuro.isna(), 'Target'] = np.nan
    df_limpo = df.dropna()

    # Salvar Dados
    df_limpo.to_csv(os.path.join(PASTA_DATA, f"{ticker.lower()}_processed.csv"))

    # 4. Treino XGBoost (FIX: Capturando as variáveis corretamente)
    model_xgb, p_xgb, acuracia_final_robusta = treinar_xgboost_multi(df_limpo, 30)
    joblib.dump(model_xgb, os.path.join(PASTA_MODELS, f"modelo_xgb_{ticker.lower()}.joblib"))

    # 5. Treino LSTM (FIX: Capturando as variáveis corretamente)
    scaler = StandardScaler()
    scaler.fit(df_limpo[['Ret_Ativo', 'Ret_Macro', 'Volatilidade']])
    joblib.dump(scaler, os.path.join(PASTA_MODELS, f"scaler_{ticker.lower()}.joblib"))
    
    model_lstm, p_lstm, acc_lstm = treinar_lstm_multi(df_limpo, 30, scaler)
    model_lstm.save(os.path.join(PASTA_MODELS, f"modelo_lstm_{ticker.lower()}.keras"))

    # 6. Salvar Metadados/Métricas (FIX: Agora as variáveis existem)
    metricas = {
        "acuracia_xgb": float(acuracia_final_robusta),
        "acuracia_lstm": float(acc_lstm),
        "data_treino": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "ticker": ticker
    }
    
    with open(os.path.join(PASTA_MODELS, f"metricas_{ticker.lower()}.json"), "w") as f:
        json.dump(metricas, f)
    
    print(f"✅ Sucesso! Modelos salvos em {PASTA_MODELS}")

if __name__ == "__main__":
    executar_pipeline_completo()