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
CAMINHO_CSV = 'hmeq.csv'
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

# SEÇÃO 2 - MODELO DE REDE NEURAL
#===================================
class RedeNeural(nn.Module):
    """
    Rede neural feedforward (MLP) para classificação binária.

    Arquitetura:
        Entrada (input_dim) -> Dense(16) -> ReLU
                           -> Dense(8)  -> ReLU
                           -> Dense(1)  -> Sigmoid -> probabilidade [0, 1]

    A saída Sigmoid retorna a probabilidade de o cliente ser inadimplente.
    Limiar de decisão: prob > 0.5 → BAD = 1 (inadimplente).
    """

    def __init__(self, input_dim: int):
        """
        Args:
            input_dim: número de features de entrada (colunas do dataset)
        """
        super(RedeNeural, self).__init__()
        self.fc1     = nn.Linear(input_dim, 16)  # Camada oculta 1
        self.fc2     = nn.Linear(16, 8)           # Camada oculta 2
        self.fc3     = nn.Linear(8, 1)            # Camada de saída
        self.sigmoid = nn.Sigmoid()               # Ativa prob. de classificação

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passagem direta (forward pass) — define como os dados fluem."""
        x = torch.relu(self.fc1(x))   # ReLU: max(0, x) — evita gradiente morto
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x)) # Comprime saída para [0, 1]
        return x
    
# SEÇÃO 3 - FUNÇÃO DE TREINO E AVALIAÇÃO
#==========================================
def treinar(modelo: nn.Module, loader: DataLoader, epocas: int) -> None:
    """
    Treina o modelo por um número fixo de épocas.

    Args:
        modelo : instância de RedeNeural
        loader : DataLoader com os dados de treino do cliente
        epocas : número de passagens completas pelo dataset local
    """
    # BCELoss = Binary Cross-Entropy, adequada para classificação binária
    criterio  = nn.BCELoss()
    otimizador = optim.Adam(modelo.parameters(), lr=LEARNING_RATE)

    modelo.train()  # Ativa modo treino (habilita dropout/batchnorm se houver)

    for epoca in range(epocas):
        for X_batch, y_batch in loader:
            otimizador.zero_grad()                           # Limpa gradientes anteriores
            predicoes = modelo(X_batch)                      # Forward pass
            loss = criterio(predicoes, y_batch.view(-1, 1)) # Calcula erro
            loss.backward()                                  # Backpropagation
            otimizador.step()                                # Atualiza pesos


def avaliar(modelo: nn.Module, loader: DataLoader) -> Tuple[float, float, float]:
    """
    Avalia o modelo sem atualizar pesos (sem gradientes).

    Returns:
        (loss_média, acurácia, f1_score)
    """
    criterio = nn.BCELoss()
    total_loss = 0.0
    corretos   = 0
    total      = 0
    todas_preds  = []
    todos_labels = []

    modelo.eval()  # Desativa dropout/batchnorm (modo avaliação)

    with torch.no_grad():  # Desativa cálculo de gradientes -> mais rápido
        for X_batch, y_batch in loader:
            saidas    = modelo(X_batch)
            total_loss += criterio(saidas, y_batch.view(-1, 1)).item()

            # Converte probabilidade em classe binária (threshold = 0.5)
            predicoes = (saidas > 0.5).float()
            total    += y_batch.size(0)
            corretos += (predicoes.view(-1) == y_batch).sum().item()

            todas_preds.extend(predicoes.view(-1).cpu().numpy())
            todos_labels.extend(y_batch.cpu().numpy())

    acuracia = corretos / total
    f1       = f1_score(todos_labels, todas_preds)
    loss_media = total_loss / len(loader)

    return loss_media, acuracia, f1

# SEÇÃO 4 — CLIENTE FLOWER (PARTICIPANTE DA FEDERAÇÃO)
# =======================================================
class ClienteHMEQ(fl.client.NumPyClient):
    """
    Representa um participante do sistema federado.

    No Aprendizado Federado cada cliente:
      1. Recebe os pesos globais do servidor (set_parameters)
      2. Treina localmente com seus dados privados (fit)
      3. Envia os novos pesos ao servidor (get_parameters)
      4. Avalia o modelo localmente e reporta métricas (evaluate)

    NumPyClient facilita a troca de pesos como arrays NumPy,
    convertendo automaticamente para tensores PyTorch.
    """

    def __init__(self, loader_treino: DataLoader, loader_teste: DataLoader, input_dim: int):
        self.loader_treino = loader_treino
        self.loader_teste  = loader_teste
        self.modelo        = RedeNeural(input_dim)

    # Serialização dos pesos
    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Extrai os pesos do modelo como lista de arrays NumPy."""
        return [val.cpu().numpy() for _, val in self.modelo.state_dict().items()]

    def set_parameters(self, parametros: List[np.ndarray]) -> None:
        """Carrega pesos (recebidos do servidor) no modelo local."""
        mapa = zip(self.modelo.state_dict().keys(), parametros)
        state_dict = {k: torch.tensor(v) for k, v in mapa}
        self.modelo.load_state_dict(state_dict, strict=True)

    # Callbacks chamados pelo servidor Flower
    def fit(self, parametros: List[np.ndarray], config: Dict):
        """
        Recebe pesos globais -> treina localmente -> devolve pesos atualizados.

        Returns:
            (pesos_atualizados, nº_amostras_treino, métricas_extras)
        """
        self.set_parameters(parametros)
        treinar(self.modelo, self.loader_treino, epocas=EPOCHS_LOCAL)
        return self.get_parameters(config={}), len(self.loader_treino.dataset), {}

    def evaluate(self, parametros: List[np.ndarray], config: Dict):
        """
        Recebe pesos globais → avalia localmente → reporta métricas ao servidor.

        Returns:
            (loss, nº_amostras_teste, dicionário_de_métricas)
        """
        self.set_parameters(parametros)
        loss, acuracia, f1 = avaliar(self.modelo, self.loader_teste)
        return float(loss), len(self.loader_teste.dataset), {
            "accuracy": float(acuracia),
            "f1": float(f1)
        }

# SEÇÃO 5 — FUNÇÃO DE AGREGAÇÃO DE MÉTRICAS
# =============================================================================

def media_ponderada(metricas: List[Tuple[int, Dict]]) -> Dict:
    """
    Agrega as métricas de todos os clientes usando média PONDERADA pelo
    número de amostras de cada cliente.

    Por que ponderada?
       Clientes com mais dados têm maior impacto na métrica global.
       Evita que um cliente com poucos dados distorça a avaliação.

    Args:
        metricas: lista de (nº_amostras, dict_métricas) por cliente

    Returns:
        Dicionário com accuracy e f1 agregados
    """
    total_amostras = sum(n for n, _ in metricas)

    acuracia_global = sum(n * m["accuracy"] for n, m in metricas) / total_amostras
    f1_global       = sum(n * m["f1"]       for n, m in metricas) / total_amostras

    return {"accuracy": acuracia_global, "f1": f1_global}

# SEÇÃO 6 — FUNÇÃO PRINCIPAL: SIMULAÇÃO FEDERADA
# =============================================================================

def main():
    """
    Orquestra toda a simulação de Aprendizado Federado:
      1. Carrega e pré-processa dados (com paralelismo)
      2. Divide os dados entre os clientes simulados
      3. Define a estratégia FedAvg
      4. Inicia a simulação com o Flower
      5. Exibe resultados finais
    """

    # Etapa 1: Dados 
    X_train, X_test, y_train, y_test = carregar_e_preprocessar_dados()
    input_dim = X_train.shape[1]  # nº de features = nº de neurônios de entrada

    # Etapa 2: Particionamento dos dados entre clientes 
    # Em cenários reais, cada cliente já teria seus próprios dados locais.
    # Aqui simulamos isso dividindo o conjunto de treino em N fatias iguais.
    X_partes = np.array_split(X_train, NUM_CLIENTES)
    y_partes = np.array_split(y_train, NUM_CLIENTES)

    print(f"Dados distribuídos entre {NUM_CLIENTES} clientes:")
    for i, (xp, yp) in enumerate(zip(X_partes, y_partes)):
        print(f"  Cliente {i}: {len(xp)} amostras de treino")

    # Etapa 3: Fábrica de clientes (client_fn) 
    # O Flower chama client_fn(cid) para instanciar cada cliente durante
    # a simulação. cid é o ID do cliente como string ("0", "1", "2", ...).
    def client_fn(cid: str) -> fl.client.Client:
        idx = int(cid)

        # Dataset local de treino (privado ao cliente idx)
        ds_treino = TensorDataset(
            torch.tensor(X_partes[idx]).float(),
            torch.tensor(y_partes[idx]).float()
        )
        # Conjunto de teste global compartilhado (para avaliação simplificada)
        ds_teste = TensorDataset(
            torch.tensor(X_test).float(),
            torch.tensor(y_test).float()
        )

        loader_treino = DataLoader(ds_treino, batch_size=BATCH_SIZE, shuffle=True)
        loader_teste  = DataLoader(ds_teste,  batch_size=BATCH_SIZE)

        # .to_client() converte NumPyClient -> Client (interface interna do Flower)
        return ClienteHMEQ(loader_treino, loader_teste, input_dim).to_client()

    # Etapa 4: Estratégia FedAvg 
    # FedAvg (McMahan et al., 2017) agrega os pesos dos clientes fazendo
    # uma média ponderada pelo número de amostras de cada cliente.
    estrategia = fl.server.strategy.FedAvg(
        fraction_fit=1.0,             # 100% dos clientes participam do treino
        min_fit_clients=NUM_CLIENTES, # Mínimo de clientes para iniciar o round
        min_available_clients=NUM_CLIENTES,
        evaluate_metrics_aggregation_fn=media_ponderada,  # Função de métricas custom
    )

    # Etapa 5: Simulação
    print(f"\n{'='*60}")
    print(f" Iniciando Aprendizado Federado")
    print(f"   Clientes : {NUM_CLIENTES}")
    print(f"   Rounds   : {NUM_ROUNDS}")
    print(f"   Épocas   : {EPOCHS_LOCAL} por round por cliente")
    print(f"{'='*60}\n")

    # client_resources limita quanto de CPU/GPU cada cliente simulado usa.
    # Evita OOM (Out of Memory) ao simular muitos clientes em paralelo.
    recursos_cliente = {"num_cpus": 1, "num_gpus": 0.0}

    historico = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTES,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=estrategia,
        client_resources=recursos_cliente,
    )

    # Etapa 6: Resultados 
    print(f"\n{'='*60}")
    print(" RESULTADOS FINAIS DA SIMULAÇÃO")
    print(f"{'='*60}")
    print(historico)

    # Exibe métricas do último round de forma mais legível
    if historico.metrics_distributed:
        ultimo_round = list(historico.metrics_distributed.items())[-1]
        round_num, metricas = ultimo_round
        print(f"\n Métricas do último round ({round_num}):")
        for nome, valor in metricas.items():
            print(f"   {nome:12s} = {valor:.4f}")

# =============================================================================
# PONTO DE ENTRADA
# =============================================================================
if __name__ == "__main__":
    main()