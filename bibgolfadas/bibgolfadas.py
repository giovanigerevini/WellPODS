# Biblioteca de funções utilizadas na análise de robustez da metodologia de ajuste 
# de modelos e controladores para sistemas dinâmicos de golfadas
# 
# aut: giovani gerevini

# Setup de importação de bibliotecas padrões
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import interp1d
from pyswarms.single import GlobalBestPSO
import numpy as np
from sklearn.preprocessing import StandardScaler

import bibgolfadas.ModeloDePredicao as ModeloDePredicao

# WELL A - ABL56
ABL56 = {
    'teta': np.pi/4,
    'ALFAgw': 0.0188,
    'mlstill': 710.9821,
    'Da': 0.140,
    'Rol': 962.9,
    'Cg': 0.00002346,
    'D': 0.1524,
    'Romres': 905.4,
    'Cout': 6.8920e-3,
    'Dt': 0.150,
    'g': 9.81,
    'Epsilon': 0.9166,
    'Hpdg': 1387,
    'M': 18.0,
    'Ka': 6.6005e-4,
    'Ht': 1433,
    'R': 8314,
    'Kw': 0.0347,
    'Hvgl': 940,
    'T': 298,
    'Kr': 691.2173,
    'L': 4497,
    'La': 940,
    'Lt': 1659,
    'Veb': 53.3003,
    'Vr': 81.601,
}

class Well:
    '''
    Classe para construção dos poços

    '''
    def __init__(self, **config):
        """
        Inicializa o objeto Well (Poço) que pode conter múltiplos modelos.
        
        Args:
            config: Configurações específicas do poço.
        """
        self.config = config
        self.modelos = {}  # Dicionário para armazenar diferentes modelos associados ao poço

    def adicionar_modelo(self, nome_modelo, modelo_tipo, **modelo_config):
        """
        Adiciona um modelo ao poço.
        
        Args:
            nome_modelo: Nome do modelo (string) para referência.
            modelo_tipo: Tipo de modelo ('FOWM', 'LSTM', 'ANN').
            modelo_config: Configurações específicas para o modelo.
        """
        if modelo_tipo == 'FOWM':
            params = modelo_config.get('params', {})
            modelo = ModeloDePredicao.FOWMModel(params=params)  # Corrige para inicializar FOWMModel corretamente com 'params'
        elif modelo_tipo == 'LSTM':
            modelo = ModeloDePredicao.LSTMModel(**modelo_config)
        elif modelo_tipo == 'ANN':
            modelo = ModeloDePredicao.ANNModel(**modelo_config)
        else:
            raise ValueError("Modelo desconhecido. Escolha entre 'FOWM', 'LSTM' ou 'ANN'.")
        
        # Constrói o modelo
        if modelo_tipo == 'FOWM':
          modelo.build(params=modelo_config.get('params', {}))  # Passa 'params' de forma consistente
        else:
            modelo.build()

        # Adiciona o modelo ao dicionário de modelos
        self.modelos[nome_modelo] = modelo

    def listar_modelos(self):
        """
        Lista todos os modelos disponíveis no poço.
        
        Returns:
            Uma lista com os nomes de todos os modelos armazenados no poço.
        """
        if not self.modelos:
            print("Nenhum modelo disponível no poço.")
        else:
            print("Modelos disponíveis no poço:")
            for nome_modelo in self.modelos:
                print(f"- {nome_modelo}")
        return list(self.modelos.keys())

    def get_modelo(self, nome_modelo):
        """
        Retorna o modelo especificado pelo nome.
        
        Args:
            nome_modelo: Nome do modelo (string).
            
        Returns:
            Modelo especificado.
        """
        modelo = self.modelos.get(nome_modelo)
        if not modelo:
            raise ValueError(f"Modelo '{nome_modelo}' não encontrado no poço.")
        return modelo

    def get_modelo_params(self, nome_modelo):
        """
        Retorna os parâmetros do modelo especificado.
        
        Args:
            nome_modelo: Nome do modelo (string).
            
        Returns:
            Dicionário com os parâmetros do modelo especificado.
        """
        modelo = self.get_modelo(nome_modelo)
        # Retorna todos os atributos do modelo como um dicionário, excluindo métodos e internos
        params = {attr: getattr(modelo, attr) for attr in dir(modelo) if not callable(getattr(modelo, attr)) and not attr.startswith("__")}
        return params

    def fit(self, nome_modelo, *args, method='PSO', objective='MSE', **kwargs):
        """
        Treina o modelo especificado usando o método de otimização definido.

        Args:
            nome_modelo: Nome do modelo (string) a ser treinado.
            method: Método de otimização ('PSO', 'TreePartizen', 'MetodoProprio').
            objective: Função objetivo ('MSE', 'custom').
            Outros argumentos específicos para o método de treinamento.
        """
        modelo = self.modelos.get(nome_modelo)
        if not modelo:
            raise ValueError(f"Modelo '{nome_modelo}' não encontrado no poço.")

        # Verifica se é um modelo FOWM que requer o método específico de otimização
        if isinstance(modelo, ModeloDePredicao.FOWMModel) and method == 'MetodoProprio':
            optimizer = ModeloDePredicao.MetodoProprio()  # Inicializa o otimizador próprio
            return modelo.train(*args, method=optimizer, objective=objective, **kwargs)
        elif isinstance(modelo, ModeloDePredicao.FOWMModel):
            return modelo.train(*args, method=method, objective=objective, **kwargs)
        else:
            # Para modelos como LSTM ou ANN, que não requerem 'method' ou 'objective'
            return modelo.train(*args, **kwargs)

    def predict(self, nome_modelo, X):
        """
        Faz predições usando o modelo treinado especificado.
        
        Args:
            nome_modelo: Nome do modelo (string) a ser usado para predição.
            X: Conjunto de características para predição.
            
        Returns:
            Predições do modelo.
        """
        modelo = self.modelos.get(nome_modelo)
        if not modelo:
            raise ValueError(f"Modelo '{nome_modelo}' não encontrado no poço.")
        return modelo.predict(X)

    def evaluate(self, nome_modelo, X_test, y_test):
        """
        Avalia o modelo especificado com base nos dados de teste fornecidos.
        
        Args:
            nome_modelo: Nome do modelo (string) a ser avaliado.
            X_test: Conjunto de características de teste.
            y_test: Conjunto de rótulos de teste.
            
        Returns:
            Métricas de avaliação do modelo.
        """
        modelo = self.modelos.get(nome_modelo)
        if not modelo:
            raise ValueError(f"Modelo '{nome_modelo}' não encontrado no poço.")
        return modelo.evaluate(X_test, y_test)

    def avaliar_todos_modelos(self, X_test, y_test):
        """
        Avalia todos os modelos associados ao poço com base nos dados de teste fornecidos.
        
        Args:
            X_test: Conjunto de características de teste.
            y_test: Conjunto de rótulos de teste.
            
        Returns:
            Um dicionário contendo as métricas de avaliação para cada modelo.
        """
        resultados = {}
        for nome_modelo, modelo in self.modelos.items():
            erro = modelo.evaluate(X_test, y_test)
            resultados[nome_modelo] = erro
        return resultados

