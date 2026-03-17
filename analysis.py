# ANÁLISE EXPLORATÓRIA - DATASET HMEQ (RISK CREDIT)
#========================================================
# Descrição:
# Script de análise exploratória de dados (EDA) aplicado ao dataset HMEQ,
# com foco em identificação de padrões, valores ausentes, estatísticas
# descritivas e análise da variável alvo (BAD).
#
# O objetivo é compreender a estrutura dos dados e preparar a base para
# etapas futuras de Machine Learning e aprendizado federado.
#========================================================

#--------------[ BIBLIOTECAS ]--------------
import pandas as pd   # Analise de dados tabelares
# ------------------------------------------

#--------------[ VARIÁVEIS GLOBAIS ]--------------
# Caminho para o DataSet
df = pd.read_csv('hmeq.csv',sep=',')
#-----------------------------------------

# FUNÇÃO PARA FORMATAR SAÍDA NO TERMINAL
#-----------------------------------------
def set_format(title=None):
    print("\n" + "="*50)

    if title:
        print(f"{title:^50}")
    print("="*50 + "\n")
#============================

# FUNÇÃO PARA EXIBIR INFORMAÇÕES GERAIS DO DATASET
#--------------------------------------------------------
def dataset_info(df):
    df.info()
#============================

# FUNÇÃO PARA ANALISAR VALORES AUSENTES
#-------------------------------------------
def out_value(df):
    # Contagem de valores nulos por coluna
    missing = df.isnull().sum()
    # Percentual de valores nulos
    percent = (df.isnull().mean()*100).round(2)
    # Criação de tabela consolidada
    result = pd.DataFrame({
        'Missing' : missing,
        '%': percent
    })
    # Exibe apenas colunas com valores ausentes ordenadas
    print(result[result['Missing'] > 0].sort_values(by='%',ascending=False))
#=======================================================================

# FUNÇÃO PARA ESTATÍSTICAS DESCRITIVAS
#-------------------------------------------
def descriptive(df):
    print(df.describe())
#==========================

# FUNÇÃO PARA DISTRIBUIÇÃO DA VARIÁVEL ALVO (BAD)
#---------------------------------------------------
def target_value(df):
    # Calcula proporção de cada classe (0 = bom, 1 = inadimplente)
    print(df['BAD'].value_counts(normalize=True))
#============================================

# FUNÇÃO PARA IDENTIFICAR COLUNAS CATEGÓRICAS
#-----------------------------------------------
def cat_cols(df):
    return df.select_dtypes(include=['object', 'string']).columns.tolist()
#====================================================

# FUNÇÃO PARA EXIBIR VALORES ÚNICOS DAS CATEGÓRICAS
#------------------------------------------------------
def unic_values(df):
    # Obtém lista de colunas categóricas
    cols = cat_cols(df)
    # Iteração sobre cada coluna
    for col in cols:
        print(f"\n Coluna: {col}")
        val = df[col].value_counts(dropna=False)
        # Remove nomes padrão do pandas para saída limpa
        val.index.name = None
        val.name = None
        print(val.to_string())
#================================

# FUNÇÃO PRINCIPAL
#-----------------------------------
if __name__ == "__main__":

    print("--- [ Informações do Dataset ] ---")
    dataset_info(df=df)
    set_format()

    print("--- [ Valores Ausentes ] ---")
    out_value(df=df)
    set_format()

    print("--- [ Estatística Descritiva ] ---")
    descriptive(df=df)
    set_format()

    print("--- [ Distribuição de Variável Alvo (BAD) ] ---")
    target_value(df=df)
    print("\nValores únicos em:\n")
    unic_values(df=df)
    set_format()
#====================================