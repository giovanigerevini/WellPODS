import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d

'''
Ferramentas de uso geral
'''

def transformar_entrada(X, freq_atual='1s', freq_destino=None, num_lags=0):
    """
    Reamostra e adiciona lags aos dados de entrada conforme especificado.
    
    Args:
        X (pd.DataFrame): DataFrame com os dados de entrada.
        freq_atual (str): Frequência atual dos dados ('1s' por padrão).
        freq_destino (str or None): Frequência desejada para reamostragem ('None' se não houver reamostragem).
        num_lags (int): Número de lags a serem adicionados (0 por padrão).
        
    Returns:
        pd.DataFrame: DataFrame transformado com reamostragem e/ou lags conforme solicitado.
    """
    X_aux = X.copy()

    # Certifique-se de que a entrada X seja um DataFrame
    if not isinstance(X_aux, pd.DataFrame):
        raise ValueError("X deve ser um pandas DataFrame.")
  
    # Verificar e converter o índice para DatetimeIndex se necessário
    if not isinstance(X_aux.index, pd.DatetimeIndex):
        X_aux.index = pd.to_datetime(X_aux.index, unit='s')  # Ajuste conforme necessário para o seu índice

    # Remover duplicatas no índice
    X_aux = X_aux[~X_aux.index.duplicated(keep='first')]

    # Inicializar df_resampled com o DataFrame original
    df_resampled = X_aux.copy()

    # Reamostragem, se especificada
    if freq_destino:
        df_resampled = X_aux.resample(freq_destino).ffill()   # Reamostra usando o último valor válido
        df_resampled = df_resampled.dropna().reset_index(drop=True)  # Remove NaNs e redefine o índice

    # Adicionando lags, se solicitado
    if num_lags > 0:
        df_base = df_resampled.copy()

        for col in df_base.columns:  # Loop através de todas as colunas
            for lag in range(1, num_lags + 1):
                df_base[f'{col}_lag_{lag}'] = df_base[col].shift(lag)

        # Remover os NaNs criados pelos lags e ajustar o tamanho
        df_base.dropna(inplace=True)

        return df_base

    # Retorna o DataFrame reamostrado se não houver lags solicitados
    return df_resampled

# Supondo que X_train e Y_train sejam seus DataFrames de entrada
def normalizar_zscore(X_train, Y_train):
    """
    Aplica a normalização Z-score usando StandardScaler aos dados de treinamento.

    Args:
        X_train (pd.DataFrame): DataFrame de características de entrada.
        Y_train (pd.DataFrame): DataFrame de variáveis de saída.

    Returns:
        tuple: DataFrames normalizados (X_train_normalizado, Y_train_normalizado).
    """
    # Instanciando o scaler
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()

    # Ajusta o scaler aos dados de treinamento e os transforma
    X_train_normalizado = scaler_X.fit_transform(X_train)
    Y_train_normalizado = scaler_Y.fit_transform(Y_train)

    # Convertendo de volta para DataFrames para manter os rótulos das colunas
    X_train_normalizado = pd.DataFrame(X_train_normalizado, columns=X_train.columns, index=X_train.index)
    Y_train_normalizado = pd.DataFrame(Y_train_normalizado, columns=Y_train.columns, index=Y_train.index)

    return X_train_normalizado, Y_train_normalizado, scaler_X, scaler_Y

def generate_dynamic_input(inputs, time_step=10):
    """
    Gera vetores de entrada dinâmica para simulação e cria funções de interpolação para cada entrada.
    
    Args:
    - inputs (list of array-like): Lista de vetores de amostras de entrada.
    - time_step (int): Passo de tempo em horas (default é 10).
    
    Returns:
    - interpolated_funcs (list of functions): Lista de funções de interpolação para cada entrada.
    - tArray_sim (numpy array): Array de tempo de simulação em segundos.
    """
    # Verifica se a entrada é uma lista e converte cada entrada para um array numpy multiplicado por 0.01
    inputs = [np.array(input_samp) * 0.01 for input_samp in inputs]
    
    # Calcula o tempo final da simulação usando o tamanho do primeiro conjunto de amostras
    tf_sim = inputs[0].size * time_step * 3600
    
    # Gera o vetor de tempo para a simulação
    tArray_sim = np.arange(0, tf_sim)
    
    # Calcula o número de repetições de cada valor de amostra
    repetitions = int(time_step * 3600)
    
    # Cria uma lista para armazenar as funções de interpolação
    interpolated_funcs = []
    
    for input_samp in inputs:
        # Repete os valores da amostra conforme necessário
        dynamic_input = np.repeat(input_samp, repetitions)
        
        # Cria a função de interpolação para a entrada atual
        interpolate_func = interp1d(tArray_sim, dynamic_input, kind='nearest', fill_value="extrapolate")
        
        # Armazena a função de interpolação na lista
        interpolated_funcs.append(interpolate_func)
    
    return interpolated_funcs, tArray_sim