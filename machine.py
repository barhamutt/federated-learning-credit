# APRENDIZADO FEDERADO COM FLOWER (fwlr) - DATASET HMEQ
#=========================================================
# O Aprendizado federado permite treinar modelos de ML de forma distribuída:
# cada "cliente" treina localmente com seus próprios dados e envia apenas os
# PESOS do modelo (não os dados brutos) para um servidor central, que agrega
# tudo via FedAvg (mnédia federada).
#
# Fluxo feral:
# 1. Servidor distribui pesos iniciais para os clientes
# 2. Cada cliente treina localmente por N épocas
# 3. Clientes enviam pesos atualizados ao servidor
# 4. Servidor agrega (FedAvg) -> repete por R rounds
#=========================================================

#---------- [ BIBLIOTECAS ] ----------
import os               # Interação com o sistema
import time             # Trabalho com tempo
import pandas as pd     # Manipulação de dados em tabelas
import numpy as np      # Calculos numericos rapidos
# Biblioteca padrão do Python para paralelismo [concurrent.futures].
# `ProcessPoolExecutor` usa processos separados (contorna o GIL) - CPU-bound
# `as_completed` retorna os resultados (futures) conforme são finalizados
from concurrent.futures import ProcessPoolExecutor, as_completed
# Biblioteca para dividir dados em treino e teste [sklearn.model_selection]
# `train_test_split` separa os dados em conjuntos (ex: 80% treino, 20% teste)
from sklearn.model_selection import train_test_split
# Ferramentas de pré-processamento [sklearn.preprocessing]
# `StandardScaler` normaliza os dados (média=0, desvio padrão=1)
# `LabelEncoder` transforma categorias (strings) em números
from sklearn.preprocessing import StandardScaler, LabelEncoder
# Métricas de avaliação de modelos [sklearn.metrics]
# accuracy_score mede a proporção de acertos
# f1_score calcula o equilíbrio entre precisão e recall (bom para dados desbalanceados)
from sklearn.metrics import accuracy_score, f1_score
import torch                 # Deep learning [PyTorch]
import torch.nn as nn        # Redes neurais 
import torch.optim as optim  # Otimizadores
# Utilidades para dados [torch.utils.data]
# `DataLoader` cria batches e embaralha dados automaticamente
# `TensorDataset` transforma tensores em dataset compatível com DataLoader
from torch.utils.data import DataLoader, TensorDataset
import flwr as fl       # Framework de aprendizado federado
# Tipagem estática do Python [typing]
# List → lista tipada; Tuple → tupla tipada;
# Dict → dicionário tipado; Optional → valor opcional.
from typing import List, Tuple, Dict, Optional
#-------------------------------------

#--------------------
# VARIAVEIS GLOBAIS
#--------------------
CAMINHO_CSV = ''
NUM_CLIENTES  = 3      # Número de participantes na federação
NUM_ROUNDS    = 10     # Rounds de comunicação clientes-servidor 
EPOCHS_LOCAL  = 5      # Epocas de treinamento local por round
BATCH_SIZE    = 32     # Tamanho do mini-batch no DataLoader
LEARNING_RATE = 0.01   # Taxa de aprendizado do Adam
TEST_SIZE     = 0.2    # Fração dos dados reservada para teste global
RANDOM_SEED   = 42     # Semente para reprodutibilidade
#--------------------

# SEÇÃO 1 - FUNÇÕES DE PRÉ-PROCESSAMENTO (COM PARALELISMO)
#=================================================
def preprocessar_coluna(args: Tuple) -> Tuple[str, pd.Series]:
    """
    Processa UMA coluna do DataFrame de forma independente.
 
    Regras:
       Coluna categórica (object) - preenche NaN com a moda e codifica com LabelEncoder
       Coluna numérica            - preenche NaN com a mediana
 
    Retorna uma tupla (nome_da_coluna, serie_processada) para que o
    ProcessPoolExecutor possa devolver os resultados fora de ordem e
    ainda assim reconstruirmos o DataFrame correto.
 
    Args:
        args: tupla (nome_coluna, série_pandas)
 
    Returns:
        (nome_coluna, série_processada)
    """
    nome_col, serie = args
 
    if serie.dtype == 'object':
        # Preenche com valor mais frequente
        serie = serie.fillna(serie.mode()[0])
        # Transforma strings em inteiros
        le = LabelEncoder()
        serie = pd.Series(le.fit_transform(serie), name=nome_col)
    else:
        # Preenche com mediana
        serie = serie.fillna(serie.median())

    return nome_col, serie
 
 
def carregar_e_preprocessar_dados(caminho: str = CAMINHO_CSV):
    """
    Lê o CSV, pré-processa as colunas em PARALELO usando ProcessPoolExecutor
    e devolve os conjuntos de treino/teste já normalizados.
 
    Por que ProcessPoolExecutor aqui?
       Cada coluna pode ser processada de forma totalmente independente.
       Em datasets largos (muitas colunas), o ganho de tempo é significativo.
       Processos separados evitam o GIL e aproveitam múltiplos núcleos da CPU.
 
    Returns:
        X_train, X_test, y_train, y_test (arrays numpy normalizados)
    """
    print("[1/4] Carregando dataset...")
    df = pd.read_csv(caminho, sep=',')
    print(f"      -> {df.shape[0]} linhas x {df.shape[1]} colunas")
 
    # ── Paralelismo: pré-processar cada coluna em um processo separado ────────
    print("[2/4] Pré-processando colunas em paralelo...")
    inicio = time.time()
 
    # Separamos a variável-alvo antes — ela não precisa de pré-processamento
    colunas_features = [c for c in df.columns if c != 'BAD']
    tarefas = [(col, df[col].copy()) for col in colunas_features]
 
    # max_workers=None - Python decide o número ideal de processos (≈ nº de CPUs)
    resultados = {}
    with ProcessPoolExecutor(max_workers=None) as executor:
        # submit() envia cada tarefa para um processo do pool
        futuros = {executor.submit(preprocessar_coluna, t): t[0] for t in tarefas}
 
        # as_completed() itera conforme os processos terminam (não necessariamente
        # na ordem de envio), permitindo coletar resultados assim que ficam prontos
        for futuro in as_completed(futuros):
            nome_col, serie_processada = futuro.result()
            resultados[nome_col] = serie_processada
 
    # Reconstrói o DataFrame mantendo a ordem original das colunas
    for col in colunas_features:
        df[col] = resultados[col]
 
    fim = time.time()
    print(f"      -> Concluído em {fim - inicio:.2f}s")
 
    # Separação features / target
    X = df[colunas_features].values    # Matriz de entrada
    y = df['BAD'].values               # Rótulo: 1 = crédito inadimplente, 0 = adimplente
 
    # Split treino / teste (estratificado para manter proporção de classes)
    print("[3/4] Dividindo em treino/teste...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y   # garante mesma proporção de BAD=0/1 em treino e teste
    )
 
    # Normalização: média 0, desvio-padrão 1
    # IMPORTANTE: fit apenas no treino para evitar "data leakage" (vazamento
    # de informação do teste para o modelo).
    print("[4/4] Normalizando features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)   # aplica a mesma escala do treino
 
    print(f"      → Treino: {X_train.shape} | Teste: {X_test.shape}\n")
    return X_train, X_test, y_train, y_test