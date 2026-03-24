# 📈 Inteligência Quantitativa: Estratégia Multivariada e IA

Um terminal financeiro interativo de nível institucional que combina análise exploratória de dados (EDA), Machine Learning quantitativo e Inteligência Artificial Generativa para prever a probabilidade direcional de ativos financeiros.

## 🚀 O que o projeto faz?
Este aplicativo não tenta prever o preço exato de uma ação amanhã (o que geralmente resulta em ruído). Em vez disso, ele converte a previsão financeira em um **problema de classificação probabilística multiclasse** (Alta, Neutro, Baixa), utilizando análise multivariada para entender a correlação entre um ativo principal (ex: QQQ) e um indexador macroeconômico (ex: ^TNX - Juros de 10 anos).

## ✨ Principais Funcionalidades

* **Análise Multivariada Dinâmica:** Sincroniza e limpa séries temporais de múltiplos ativos, tratando automaticamente fusos horários e feriados distintos.
* **Engenharia de Features Avançada:** Calcula indicadores técnicos críticos como Lags de retorno, Distância de Médias Móveis (SMAs), Bandas de Bollinger e Volatilidade Rolante.
* **Validação Cruzada (Walk-Forward):** Utiliza `TimeSeriesSplit` no modelo XGBoost para evitar *Data Leakage* (vazamento de dados) e simular testes no mundo real em diferentes regimes de mercado.
* **Deep Learning (Tensores 3D):** Implementa uma rede neural LSTM que lê matrizes temporais (janelas de 60 dias) com múltiplos canais de dados (Retorno do Ativo, Retorno Macro e Volatilidade).
* **Agente Autônomo (Gemini 2.0 Flash):** Integra a API do Google Gemini para atuar como um Gestor de Portfólio, interpretando as probabilidades matemáticas e redigindo um relatório executivo com sugestões de alocação de capital.
* **Dashboard Interativo:** Visualizações financeiras avançadas criadas com Plotly (Gráficos de Velas OHLC, Histogramas com SMAs sobrepostas e Box Plots).

## 🛠️ Como executar localmente

1. Clone o repositório.
2. Crie um ambiente virtual (Python 3.11 recomendado).
3. Instale as dependências: `pip install -r requirements.txt`
4. Execute o terminal: `streamlit run app_financeiro.py`