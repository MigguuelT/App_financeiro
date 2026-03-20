# Análise de Preços e Previsão com Machine Learning 📈🤖

Este aplicativo web, construído em Python utilizando a framework **Streamlit**, tem o objetivo de analisar o histórico de contratos e ativos financeiros (como o Cobre `HG=F`, Ações, etc.), calcular métricas de avaliação e projetar preços para o curto prazo utilizando **Machine Learning (XGBoost)**.

Além da predição estatística, a aplicação integra a **Gemini API** com capacidade de busca na web (Search Grounding) em tempo real, atuando como um Agente Financeiro que cruza as predições matemáticas com o atual cenário geopolítico e macroeconômico mundial.

## 🚀 Funcionalidades
* **Extração de Dados:** Coleta automatizada de histórico de preços via `yfinance`.
* **Avaliação de Modelos:** Divisão dos dados em treino/teste com cálculo automático de `MAE`, `RMSE` e `R²`.
* **Predição Futura:** Projeção autorregressiva dos próximos 30 a 90 dias.
* **Visualização Interativa:** Gráficos dinâmicos com a biblioteca Plotly.
* **Agente Financeiro IA:** Geração de relatórios descritivos contextualizados com as manchetes do dia.

## 🛠️ Como executar localmente
1. Clone este repositório no seu ambiente local.
2. Certifique-se de ter o Python instalado. Recomenda-se criar um ambiente virtual isolado.
3. Instale as dependências executando o comando via pip:
   ```bash
   pip install -r requirements.txt