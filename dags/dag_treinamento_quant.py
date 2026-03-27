from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os

# --- LÓGICA DE CAMINHO PARA O AIRFLOW ---
# 1. Identifica onde a DAG está (pasta /dags)
DIR_DAG = os.path.dirname(os.path.abspath(__file__))
# 2. Sobe um nível para a raiz do projeto
RAIZ_PROJETO = os.path.abspath(os.path.join(DIR_DAG, ".."))
# 3. Adiciona a pasta /scripts ao sistema para o import funcionar
sys.path.append(os.path.join(RAIZ_PROJETO, "scripts"))

# Agora o import é seguro, pois o sistema conhece a pasta
from pipeline_treino import executar_pipeline_completo

default_args = {
    'owner': 'Mig',
    'depends_on_past': False,
    'start_date': datetime(2026, 3, 20),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'dag_treinamento_quant_final',
    default_args=default_args,
    description='Pipeline robusto com caminhos dinâmicos para MLOps',
    schedule_interval=None, # Mantido manual para seus testes iniciais
    catchup=False,
    tags=['producao', 'quant']
) as dag:

    task_treino = PythonOperator(
        task_id='treinar_modelos_quant',
        python_callable=executar_pipeline_completo,
        op_kwargs={'ticker': 'QQQ', 'macro': '^TNX'}
    )

    task_treino