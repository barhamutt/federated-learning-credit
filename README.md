# 🏦 Federated Learning Credit — HMEQ Dataset

 Simulação de Aprendizado Federado aplicado à análise de risco de crédito, utilizando o framework Flower (flwr) e PyTorch.

---

## 📋 Sobre o Projeto

Este projeto implementa um sistema de **Aprendizado Federado** (*Federated Learning*) para prever inadimplência de clientes com base no dataset **HMEQ** (*Home Equity*). O modelo é treinado de forma distribuída entre múltiplos clientes simulados, sem que os dados brutos sejam compartilhados — apenas os pesos da rede neural são enviados ao servidor central para agregação via **FedAvg**.

O repositório também inclui um script completo de **Análise Exploratória de Dados (EDA)** com detecção de outliers, análise de valores ausentes e correlação com a variável alvo.

---

## 🗂️ Estrutura do Projeto

```
federated-learning-credit/
├── hmeq.csv              # Dataset de risco de crédito
├── machine_learning.py   # Simulação de aprendizado federado
├── data_analysis.py      # Análise exploratória de dados 
└── README.md
```

---

## 📊 Sobre o Dataset

O dataset **HMEQ** contém informações sobre solicitações de empréstimo com garantia de imóvel. A variável alvo é `BAD`:

| Valor | Significado |
|---|---|
| `0` | Adimplente (pagou o empréstimo) |
| `1` | Inadimplente (não pagou) |

| Coluna | Descrição |
|---|---|
| `LOAN` | Valor solicitado do empréstimo |
| `MORTDUE` | Valor devido na hipoteca existente |
| `VALUE` | Valor atual do imóvel |
| `REASON` | Motivo do empréstimo (DebtCon / HomeImp) |
| `JOB` | Categoria profissional |
| `YOJ` | Anos no emprego atual |
| `DEROG` | Número de registros depreciativos |
| `DELINQ` | Número de linhas de crédito em atraso |
| `CLAGE` | Idade da linha de crédito mais antiga (meses) |
| `NINQ` | Número de consultas de crédito recentes |
| `CLNO` | Número de linhas de crédito |
| `DEBTINC` | Relação dívida/renda |

---

## ⚙️ Pré-requisitos

- Ubuntu 24.04+ (ou similar)
- Python 3.12+

### Instalação das dependências

```bash
# Atualizar o pip (boa prática)
pip install --upgrade pip --break-system-packages
# Protobuf fixado (obrigatorio antes do flower)
pip install "protobuf==3.20.3" --break-system-packages
# Flower com suporte a simulações 
pip install -U "flwr[simulation]" --break-system-packages
# Instalar PyTorch (CPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --break-system-packages
# Bibliotecas de dados e ML
pip install pandas numpy scikit-learn --break-system-packages
```

---

## ▶️ Como Executar

### Análise Exploratória

```bash
python3 data_analysis.py
```

Exibe no terminal:
- Informações gerais do dataset
- Análise de valores ausentes com classificação de severidade
- Estatísticas descritivas com assimetria (skew)
- Distribuição e desbalanceamento da variável alvo
- Frequência das variáveis categóricas
- Detecção de outliers pelo método IQR
- Correlação de Pearson com a variável alvo

### Aprendizado Federado

```bash
python3 machine_learning.py
```

Executa a simulação com 3 clientes federados por 10 rounds de comunicação, exibindo loss, acurácia e F1-Score por round.

---

## 🧠 Arquitetura do Modelo

Rede neural *feedforward* (MLP) para classificação binária:

```
Entrada (12 features)
      ↓
  Dense(16) → ReLU
      ↓
  Dense(8)  → ReLU
      ↓
  Dense(1)  → Sigmoid → P(inadimplente)
```

Limiar de decisão: `prob > 0.5 → BAD = 1`

---

## 🔁 Fluxo do Aprendizado Federado

```
1. Servidor distribui pesos iniciais aos clientes
2. Cada cliente treina localmente (5 épocas) com seus dados privados
3. Clientes enviam apenas os pesos atualizados ao servidor
4. Servidor agrega via FedAvg (média ponderada por nº de amostras)
5. Repete por 10 rounds
```

---

## 📈 Resultados

Simulação com **3 clientes**, **10 rounds**, **5 épocas locais** por round:

| Round | Acurácia | F1-Score |
|---|---|---|
| 1  | 83.97% | 0.482 |
| 5  | 87.50% | 0.589 |
| 10 | **88.00%** | **0.632** |


---

## 🛠️ Tecnologias Utilizadas

- [Ubuntu 24.04](https://ubuntu.com/) - Sistema Operacional
- [Python 3.12](https://www.python.org/) - Linguagem de Programação
- [Flower (flwr)](https://flower.ai/) - Framework de Aprendizado Federado
- [PyTorch](https://pytorch.org/) - Rede neural e treinamento
- [scikit-learn](https://scikit-learn.org/) - Pré-processamento e métricas
- [pandas](https://pandas.pydata.org/) - Manipulação de dados
- [NumPy](https://numpy.org/) - Operações numéricas
- [concurrent.futures](https://docs.python.org/3/library/concurrent.futures.html) - Paralelismo no pré-processamento

---

## 👥 Autores

| Nome | GitHub |
|---|---|
| Ian Lucas Lobato Barra da Silva | [@barhamutt](https://github.com/barhamutt) |
| Ilard Moisés da Silva Lamarão | [@IlardLamarao](https://github.com/IlardLamarao) |

---

<p align="center">
  Desenvolvido como Projeto Integrador das disciplinas de Inteligência Artificial 1, Programação Paralela, Ciência de dados e Laboratorio de Programação Científica. <br> - IFPA (6° semestre - Bacharelado de Ciência da Computação).
</p>