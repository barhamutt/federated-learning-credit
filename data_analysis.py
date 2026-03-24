# ANÁLISE EXPLORATÓRIA - DATASET HMEQ (RISK CREDIT)
#========================================================
# Descrição:
# Script de análise exploratória de dados (EDA) aplicado ao dataset HMEQ,
# com foco em identificação de padrões, valores ausentes, estatísticas
# descritivas e análise da variável alvo (BAD).
#
# O objetivo é compreender a estrutura dos dados e preparar a base para
# etapas futuras de Machine Learning e aprendizado federado.
#
# Colunas do dataset:
#   BAD     - Variável alvo: 1 = inadimplente, 0 = adimplente
#   LOAN    - Valor solicitado do empréstimo
#   MORTDUE - Valor devido na hipoteca existente
#   VALUE   - Valor atual do imóvel
#   REASON  - Motivo do empréstimo (DebtCon / HomeImp)
#   JOB     - Categoria profissional
#   YOJ     - Anos no emprego atual
#   DEROG   - Número de registros depreciativos
#   DELINQ  - Número de linhas de crédito em atraso
#   CLAGE   - Idade da linha de crédito mais antiga (meses)
#   NINQ    - Número de consultas de crédito recentes
#   CLNO    - Número de linhas de crédito
#   DEBTINC - Relação dívida/renda (debt-to-income ratio)
#========================================================

#--------------[ BIBLIOTECAS ]--------------
import pandas as pd   # Análise de dados tabelares
import numpy as np    # Cálculos numéricos
#-------------------------------------------

#--------------[ CORES ANSI ]--------------
# Sequências de escape ANSI para colorir texto no terminal Linux/Ubuntu.
# \033[ inicia o código de escape; o número define a cor; m encerra.
# RESET volta ao padrão do terminal após aplicar a cor.
VERMELHO = '\033[91m'
AMARELO  = '\033[93m'
VERDE    = '\033[92m'
RESET    = '\033[0m'
#------------------------------------------

#--------------[ VARIÁVEIS GLOBAIS ]--------------
df = pd.read_csv('hmeq.csv', sep=',')       # Caminho para modelo CSV que será analisado
# Limiar para classificar missing como "crítico"
LIMITE_MISSING_CRITICO = 10.0  # %
# Limiar IQR para detecção de outliers (padrão estatístico)
FATOR_IQR = 1.5
#-------------------------------------------------

# UTILITÁRIO: FORMATAÇÃO DE SAÍDA NO TERMINAL
#===============================================
def set_format(title=None):
    print("\n" + "=" * 60)
    if title:
        print(f"{title:^60}")
    print("=" * 60 + "\n")

# SEÇÃO 1 - INFORMAÇÕES GERAIS DO DATASET
#===========================================
def dataset_info(df: pd.DataFrame) -> None:
    """
    Exibe estrutura geral: tipos de dados, contagem de nulos e uso de memória.
    Útil para ter uma visão rápida do que existe no dataset.
    """
    print(f"  Linhas   : {df.shape[0]}")
    print(f"  Colunas  : {df.shape[1]}")
    print(f"  Duplicatas: {df.duplicated().sum()}")
    print()
    df.info()

# SEÇÃO 2 - ANÁLISE DE VALORES AUSENTES (MISSING VALUES)
#==========================================================
def out_value(df: pd.DataFrame) -> None:
    """
    Analisa a quantidade e proporção de valores ausentes por coluna.

    Por que isso importa?
      Valores ausentes podem distorcer o modelo se não forem tratados.
      Colunas com >10% de missing exigem atenção especial: pode-se
      imputar (preencher com média/mediana/moda) ou, em casos extremos,
      remover a coluna se ela tiver >50% de ausência.

    Classificação adotada:
      [OK]      - menos de 5% ausente
      [ATENÇÃO] - entre 5% e 10% ausente
      [CRÍTICO] - mais de 10% ausente (impacto direto no modelo)
    """
    missing  = df.isnull().sum()
    percent  = (df.isnull().mean() * 100).round(2)

    result = pd.DataFrame({
        'Ausentes' : missing,
        '%'        : percent,
    })
    result = result[result['Ausentes'] > 0].sort_values(by='%', ascending=False)

    # Adiciona classificação de severidade com cores ANSI
    def classificar(p):
        if p >= LIMITE_MISSING_CRITICO:
            return f'{VERMELHO}CRÍTICO{RESET}'
        elif p >= 5:
            return f'{AMARELO}ATENÇÃO{RESET}'
        else:
            return f'{VERDE}OK{RESET}'

    result['Severidade'] = result['%'].apply(classificar)

    print(result.to_string())

    # Resumo rápido
    criticos = result[result['%'] >= LIMITE_MISSING_CRITICO]
    if not criticos.empty:
        print(f"\n  [!]  {len(criticos)} coluna(s) com missing CRÍTICO (>={LIMITE_MISSING_CRITICO}%):")
        for col, row in criticos.iterrows():
            print(f"     • {col}: {row['%']}% ausente → considere imputação ou exclusão")

    total_celulas   = df.shape[0] * df.shape[1]
    total_ausentes  = df.isnull().sum().sum()
    print(f"\n  Total de células ausentes: {total_ausentes} / {total_celulas} "
          f"({total_ausentes/total_celulas*100:.2f}% do dataset)")

# SEÇÃO 3 - ESTATÍSTICAS DESCRITIVAS
# ======================================
def descriptive(df: pd.DataFrame) -> None:
    """
    Estatísticas descritivas das variáveis numéricas.

    Métricas exibidas:
      count - quantas linhas têm valor (sem nulos)
      mean  - média aritmética
      std   - desvio padrão (dispersão dos dados)
      min / max - valores extremos
      25%, 50%, 75% - quartis (50% = mediana)

    Quando mean >> median (50%), há assimetria positiva (outliers altos).
    """
    numericas = df.select_dtypes(include='number')
    stats = numericas.describe().T  # .T = transposto, mais legível

    # Adiciona mediana separada para facilitar comparação com média
    stats['median'] = numericas.median()
    stats['skew']   = numericas.skew().round(3)

    # Reorganiza colunas
    stats = stats[['count', 'mean', 'median', 'std', 'min', '25%', '75%', 'max', 'skew']]
    print(stats.to_string())
    print("\n  skew > 1 ou < -1 indica forte assimetria (presença de outliers)")

# SEÇÃO 4 - DISTRIBUIÇÃO DA VARIÁVEL ALVO (BAD)
# ================================================
def target_value(df: pd.DataFrame) -> None:
    """
    Analisa o balanceamento da variável alvo BAD.

    Desbalanceamento é comum em dados de crédito: a maioria dos clientes
    paga em dia (BAD=0), então o modelo pode 'aprender' a sempre prever 0
    e ainda ter alta acurácia - o F1-Score é mais honesto nesse contexto.
    """
    contagem    = df['BAD'].value_counts()
    proporcao   = df['BAD'].value_counts(normalize=True).mul(100).round(2)

    tabela = pd.DataFrame({
        'Contagem'   : contagem,
        'Proporção %': proporcao
    })
    tabela.index = tabela.index.map({0: '0 - Adimplente', 1: '1 - Inadimplente'})
    print(tabela.to_string())

    razao = contagem[0] / contagem[1]
    print(f"\n  Razão de desbalanceamento: {razao:.1f}:1  (adimplente:inadimplente)")

    if razao > 3:
        print("  [!]  Dataset desbalanceado — considere técnicas como SMOTE ou "
              "ajuste de class_weight no modelo.")

# SEÇÃO 5 - VARIÁVEIS CATEGÓRICAS
# ==================================
def cat_cols(df: pd.DataFrame) -> list:
    """Retorna lista de colunas categóricas (tipo object ou string)."""
    return df.select_dtypes(include=['object', 'string']).columns.tolist()


def unic_values(df: pd.DataFrame) -> None:
    """
    Exibe a frequência de cada categoria nas colunas categóricas.
    Inclui NaN para visualizar missing dentro das categorias.
    """
    cols = cat_cols(df)
    for col in cols:
        print(f"\n  Coluna: {col}")
        val = df[col].value_counts(dropna=False)
        val.index.name = None
        val.name       = None
        total = val.sum()
        # Adiciona percentual ao lado da contagem
        pct = (val / total * 100).round(2)
        tabela = pd.DataFrame({'Contagem': val, '%': pct})
        print(tabela.to_string())

# SEÇÃO 6 - DETECÇÃO DE OUTLIERS (MÉTODO IQR)
# ==============================================
def detectar_outliers(df: pd.DataFrame) -> None:
    """
    Identifica outliers nas colunas numéricas usando o método IQR.

    Método IQR (Intervalo Interquartil):
      Q1 = 25º percentil, Q3 = 75º percentil
      IQR = Q3 - Q1
      Limite inferior = Q1 - 1.5 * IQR
      Limite superior = Q3 + 1.5 * IQR
      Valores fora desses limites são considerados outliers.

    Outliers em dados de crédito podem ser legítimos (ex: alguém com
    dívida muito alta) ou erros de entrada. Devem ser investigados antes
    de remover.
    """
    numericas = df.select_dtypes(include='number').drop(columns=['BAD'], errors='ignore')
    resultado = []

    for col in numericas.columns:
        serie = df[col].dropna()
        Q1, Q3 = serie.quantile(0.25), serie.quantile(0.75)
        IQR    = Q3 - Q1
        lim_inf = Q1 - FATOR_IQR * IQR
        lim_sup = Q3 + FATOR_IQR * IQR

        n_outliers = ((serie < lim_inf) | (serie > lim_sup)).sum()
        pct        = round(n_outliers / len(serie) * 100, 2)

        resultado.append({
            'Coluna'   : col,
            'Outliers' : n_outliers,
            '%'        : pct,
            'Lim. Inf' : round(lim_inf, 2),
            'Lim. Sup' : round(lim_sup, 2),
        })

    tabela = pd.DataFrame(resultado).set_index('Coluna')
    tabela = tabela.sort_values('%', ascending=False)
    print(tabela.to_string())
    print(f"\n  Método: IQR x {FATOR_IQR}  |  Colunas com outliers > 5%: "
          f"{(tabela['%'] > 5).sum()}")

# SEÇÃO 7 - CORRELAÇÃO COM A VARIÁVEL ALVO
# ============================================
def correlacao_com_alvo(df: pd.DataFrame) -> None:
    """
    Calcula a correlação de Pearson entre cada feature numérica e BAD.

    Correlação de Pearson vai de -1 a +1:
      +1 -> relação positiva perfeita (feature sobe, BAD sobe)
      -1 -> relação negativa perfeita
       0 -> sem relação linear

    Valores absolutos > 0.1 já são relevantes em datasets financeiros.
    Features com correlação próxima de 0 podem ter baixo poder preditivo.

    Limitação: Pearson só captura relações LINEARES. Features não-lineares
    podem ainda ser úteis ao modelo mesmo com baixa correlação aqui.
    """
    numericas  = df.select_dtypes(include='number')
    correlacao = numericas.corr()['BAD'].drop('BAD').sort_values(key=abs, ascending=False)

    tabela = pd.DataFrame({
        'Correlação c/ BAD': correlacao.round(4),
        'Força': correlacao.abs().apply(
            lambda x: f'{VERMELHO}Alta{RESET}' if x > 0.3 else (f'{AMARELO}Moderada{RESET}' if x > 0.1 else f'{VERDE}Baixa{RESET}')
        )
    })
    print(tabela.to_string())
    print("\n  Referência: |r| > 0.3 = alta  |  0.1–0.3 = moderada  |  < 0.1 = baixa")

# FUNÇÃO PRINCIPAL
# =======================
if __name__ == "__main__":

    set_format("INFORMAÇÕES GERAIS DO DATASET")
    dataset_info(df=df)

    set_format("ANÁLISE DE VALORES AUSENTES")
    out_value(df=df)

    set_format("ESTATÍSTICAS DESCRITIVAS")
    descriptive(df=df)

    set_format("DISTRIBUIÇÃO DA VARIÁVEL ALVO (BAD)")
    target_value(df=df)

    set_format("VARIÁVEIS CATEGÓRICAS")
    unic_values(df=df)

    set_format("DETECÇÃO DE OUTLIERS (MÉTODO IQR)")
    detectar_outliers(df=df)

    set_format("CORRELAÇÃO COM A VARIÁVEL ALVO (BAD)")
    correlacao_com_alvo(df=df)
    print("\n\n") #FIM DA ANÁLISE EXPLORATÓRIA