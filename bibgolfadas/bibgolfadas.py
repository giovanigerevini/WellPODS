# Biblioteca de funções utilizadas na análise de robustez da metodologia de ajuste 
# de modelos e controladores para sistemas dinâmicos de golfadas
# 
# aut: giovani gerevini

# Setup de importação de bibliotecas padrões
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint # Import odeint
from scipy import stats
from scipy.interpolate import interp1d
import tensorflow as tf
from pyswarms.single import GlobalBestPSO
import numpy as np
from sklearn.preprocessing import StandardScaler


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
    'E': 0.9166,
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

from sklearn.preprocessing import StandardScaler
import pandas as pd

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

class Well:
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
            modelo = FOWMModel(params=params)  # Corrige para inicializar FOWMModel corretamente com 'params'
        elif modelo_tipo == 'LSTM':
            modelo = LSTMModel(**modelo_config)
        elif modelo_tipo == 'ANN':
            modelo = ANNModel(**modelo_config)
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
        if isinstance(modelo, FOWMModel) and method == 'MetodoProprio':
            optimizer = MetodoProprio()  # Inicializa o otimizador próprio
            return modelo.train(*args, method=optimizer, objective=objective, **kwargs)
        elif isinstance(modelo, FOWMModel):
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

class FOWM:
    def __init__(self, params):
        """
        Inicializa a classe FOWM com os parâmetros fornecidos.
        
        Args:
            params: Dicionário contendo os parâmetros específicos do poço.
        """
        # Definindo os atributos diretamente a partir dos parâmetros
        self.ssp0 = [1000, 1000, 1000, 1000, 1000, 1000]  # Estado inicial padrão

        for key, value in params.items():
            setattr(self, key, value)

        # Parâmetros de ajuste obrigatórios
        self.required_adjust_params = ['mlstill', 'Cg', 'Cout', 'Veb', 'E', 'Kw', 'Ka', 'Kr', 'Vr']
        
        # Inicializa os parâmetros de ajuste como None se não forem fornecidos
        for param in self.required_adjust_params:
            if not hasattr(self, param):
                setattr(self, param, None)

        # Atributo para indicar se o modelo foi treinado (parâmetros definidos)
        self.trained = False

        # Verifica se todos os parâmetros de ajuste foram fornecidos e define o estado de treinamento
        self.check_required_adjust_params()

        # Calcula as constantes e atributos derivados
        self.A = np.pi * self.D ** 2 / 4
        self.Vt = self.Lt * np.pi * self.Dt ** 2 / 4
        self.Va = self.La * np.pi * self.Da ** 2 / 4
        self.Vr = self.L * np.pi * self.D ** 2 / 4
        self.fat_Wgc = 101325 * self.M / (293 * self.R) / 3600 / 24
        self.RT = self.R * self.T
        self.RT_over_M = self.R * self.T / self.M
        self.g_sin_teta_over_A = self.g * np.sin(self.teta) / self.A
        self.g_Hvgl = self.g * self.Hvgl
        self.g_Hvgl_over_Vt = self.g * self.Hvgl / self.Vt
        self.Romres_g_deltaGlPdg = self.Romres * self.g * (self.Hpdg - self.Hvgl)
        self.Romres_g_deltaWellPdg = self.Romres * self.g * (self.Ht - self.Hpdg)
        self.anular_pressure = self.R * self.T / self.M / self.Va + self.g * self.La / self.Va
        
    def check_required_adjust_params(self):
        """
        Verifica se todos os parâmetros de ajuste obrigatórios foram definidos.
        Se todos estiverem definidos, define o atributo 'trained' como True.
        Caso contrário, define 'trained' como False e gera um erro.
        """
        undefined_params = [param for param in self.required_adjust_params if getattr(self, param) is None]
        if undefined_params:
            self.trained = False
            raise ValueError(f"Parâmetros de ajuste não definidos: {', '.join(undefined_params)}.")
        else:
            self.trained = True

    def set_params(self, **kwargs):
        """
        Define ou atualiza os parâmetros de ajuste do modelo.
        
        Args:
            kwargs: Dicionário com os parâmetros de ajuste.
        """
        for param, value in kwargs.items():
            if param in self.required_adjust_params:
                setattr(self, param, value)
        self.check_required_adjust_params()  # Verifica e atualiza o estado de 'trained' após definir os parâmetros

    def predict_y(self, t, ssp, z, Wgc, Pr, Ps):
        """
        """
        all_vars = {
            'z': [],
            'Wgc': [],
            'Pr': [],
            'Ps': [],
            'x1t': [],
            'x2t': [],
            'x3t': [],
            'x4t': [],
            'x5t': [],
            'x6t': [],
            'Peb': [],
            'Prt': [],
            'Prb': [],
            'ALFAg': [],
            'Wout': [],
            'Wg': [],
            'Vgt': [],
            'ROgt': [],
            'ROmt': [],
            'Ptt': [],
            'Ptb': [],
            'Ppdg': [],
            'Pbh': [],
            'ALFAgt': [],
            'Wwh': [],
            'Wr': [],
            'Pai': [],
            'ROai': [],
            'Wiv': [],
            'Wlout': [],
            'Wgout': [],
            't': t,
        }

        # Preenchendo as variáveis iniciais
        all_vars['z'] = np.ones(t.shape[0]) * z if isinstance(z, (int, float)) else z
        all_vars['Wgc'] = np.ones(t.shape[0]) * Wgc if isinstance(Wgc, (int, float)) else Wgc
        all_vars['Pr'] = np.ones(t.shape[0]) * Pr if isinstance(Pr, (int, float)) else Pr
        all_vars['Ps'] = np.ones(t.shape[0]) * Ps if isinstance(Ps, (int, float)) else Ps

        try:
            all_vars['x1t'], all_vars['x2t'], all_vars['x3t'], all_vars['x4t'], all_vars['x5t'], all_vars['x6t'] = (
                ssp[:, 0], ssp[:, 1], ssp[:, 2], ssp[:, 3], ssp[:, 4], ssp[:, 5]
            )
        except:
            all_vars['x1t'], all_vars['x2t'], all_vars['x3t'], all_vars['x4t'], all_vars['x5t'], all_vars['x6t'] = (
                [ssp[0]] * t.shape[0], [ssp[1]] * t.shape[0], [ssp[2]] * t.shape[0], [ssp[3]] * t.shape[0], [ssp[4]] * t.shape[0], [ssp[5]] * t.shape[0]
            )

        # Calculando as variáveis internas e armazenando nos dicionários
        all_vars['Peb'] = all_vars['x1t'] * self.RT_over_M / self.Veb
        all_vars['Prt'] = all_vars['x2t'] * self.RT_over_M / (self.Vr - (all_vars['x3t'] + self.mlstill) / self.Rol)
        all_vars['Prb'] = all_vars['Prt'] + (all_vars['x3t'] + self.mlstill) * self.g_sin_teta_over_A
        all_vars['ALFAg'] = all_vars['x2t'] / (all_vars['x2t'] + all_vars['x3t'])
        all_vars['Wout'] = [
            self.Cout * all_vars['z'][i] * np.sqrt(self.Rol * max(0, (all_vars['Prt'][i] - all_vars['Ps'][i])))
            for i in range(len(all_vars['z']))
        ]
        all_vars['Wg'] = self.Cg * np.maximum(0, (all_vars['Peb'] - all_vars['Prb']))
        all_vars['Vgt'] = self.Vt - all_vars['x6t'] / self.Rol
        all_vars['ROgt'] = all_vars['x5t'] / all_vars['Vgt']
        all_vars['ROmt'] = (all_vars['x5t'] + all_vars['x6t']) / self.Vt
        all_vars['Ptt'] = all_vars['ROgt'] * self.RT_over_M
        all_vars['Ptb'] = all_vars['Ptt'] + all_vars['ROmt'] * self.g * self.Hvgl
        all_vars['Ppdg'] = all_vars['Ptb'] + self.Romres_g_deltaGlPdg
        all_vars['Pbh'] = all_vars['Ppdg'] + self.Romres_g_deltaWellPdg
        all_vars['ALFAgt'] = all_vars['x5t'] / (all_vars['x5t'] + all_vars['x6t'])
        all_vars['Wwh'] = self.Kw * np.sqrt(self.Rol * np.maximum(0, (all_vars['Ptt'] - all_vars['Prb'])))
        all_vars['Wr'] = self.Kr * (1 - 0.2 * all_vars['Pbh'] / all_vars['Pr'] - 0.8 * (all_vars['Pbh'] / all_vars['Pr']) ** 2)
        all_vars['Pai'] = self.anular_pressure * all_vars['x4t']
        all_vars['ROai'] = all_vars['Pai'] / self.RT_over_M
        all_vars['Wiv'] = self.Ka * np.sqrt(np.maximum(0, (all_vars['Pai'] - all_vars['Ptb'])))
        all_vars['Wlout'] = (1 - all_vars['ALFAg']) * all_vars['Wout']
        all_vars['Wgout'] = all_vars['ALFAg'] * all_vars['Wout']
        
        return all_vars

    def predict(self, X, internal_vars_requested=None):
        """
        Predict the system's response using the optimized parameters and initial conditions.
        
        Args:
            ssp0: Initial state space vector
            tArray_sim: Time array for simulation
            z_sim: Choke input array
            Wgc_sim: Gas lift flow input array
            Pr_sim: Reservoir pressure array
            Ps_sim: Production header pressure array
            internal_vars_requested: List of internal variable names to return

        Returns:
            x_solution: Integrated solution over the time array
            internal_vars: Dictionary containing only the requested internal variables over time
        """
        # Verifica se todos os parâmetros de ajuste foram definidos
        self.check_required_adjust_params()

        tArray_sim, z_func, Wgc_sim, Pr_sim, Ps_sim = X

        if internal_vars_requested is None:
            internal_vars_requested = []


        def Velocity_disturb_collect(ssp, t, z_func, Wgc, Pr, Ps):
            # Read state space points
            x1, x2, x3, x4, x5, x6 = ssp  
            # Obter o valor de z no tempo t
            z = z_func(t)
            
            # Cálculos das variáveis internas (como no Velocity_disturb)
            Peb   = x1 * self.RT_over_M / self.Veb
            Prt   = x2 * self.RT_over_M / (self.Vr - (x3 + self.mlstill) / self.Rol)
            Prb   = Prt + (x3 + self.mlstill) * self.g_sin_teta_over_A
            ALFAg = x2 / (x2 + x3)
            Wout  = self.Cout * z * np.sqrt((self.Rol * max(0, (Prt - Ps))))
            Wg    = self.Cg * max(0, (Peb - Prb))
            Vgt   = self.Vt - x6 / self.Rol
            ROgt  = x5 / Vgt
            ROmt  = (x5 + x6) / self.Vt
            Ptt   = ROgt * self.RT_over_M
            Ptb   = Ptt + ROmt * self.g_Hvgl
            Ppdg  = Ptb + self.Romres_g_deltaGlPdg
            Pbh   = Ppdg + self.Romres_g_deltaWellPdg
            ALFAgt = x5 / (x5 + x6)
            Wwh   = self.Kw * np.sqrt(self.Rol * max(0, (Ptt - Prb)))
            Wr_pre = self.Kr * (1 - 0.2 * Pbh / Pr - 0.8 * (Pbh / Pr) ** 2)
            Wr    = max(0, Wr_pre)
            Pai  = self.anular_pressure * x4
            ROai = Pai / self.RT_over_M
            Wiv  = self.Ka * np.sqrt(ROai * max(0, (Pai - Ptb)))
            
            # Retorno dos valores do sistema dinâmico
            dx1 = (1 - self.E) * ALFAgt * Wwh - Wg
            dx2 = self.E * Wwh * ALFAgt + Wg - ALFAg * Wout
            dx3 = Wwh * (1 - ALFAgt) - (1 - ALFAg) * Wout
            dx4 = Wgc - Wiv
            dx5 = Wr * self.ALFAgw + Wiv - Wwh * ALFAgt
            dx6 = Wr * (1 - self.ALFAgw) - Wwh * (1 - ALFAgt) 
            
            vel = np.array([dx1, dx2, dx3, dx4, dx5, dx6], float)
            return vel

        # Solução da EDO com coleta de variáveis internas
        x_solution = odeint(
            Velocity_disturb_collect,
            self.ssp0, 
            tArray_sim, 
            args=(z_func, Wgc_sim, Pr_sim, Ps_sim),
            rtol=1e-9,
            atol=1e-11
        )
 
        if internal_vars_requested:
            all_internal_vars = self.predict_y(
                tArray_sim,
                x_solution,
                z_func(tArray_sim),
                Wgc_sim,
                Pr_sim,
                Ps_sim
            )

        # Filtrar apenas as variáveis internas solicitadas
        internal_vars = {var: all_internal_vars[var] for var in internal_vars_requested if var in all_internal_vars}

        return x_solution, internal_vars

    def bifurcation():
        pass

from abc import ABC, abstractmethod

class ModeloDePredicao(ABC):
    def __init__(self, **config):
        """
        Inicializa o modelo com a configuração fornecida.
        
        Args:
            config: Dicionário ou argumentos que definem as configurações do modelo, 
                    incluindo hiperparâmetros, tipo de modelo, etc.
        """
        self.config = config
        self.model = None  # O modelo subjacente, pode ser um objeto de ML ou modelo físico
        self.trained = False  # Flag para indicar se o modelo foi treinado
        self.history = None  # Histórico de treinamento e métricas

    @abstractmethod
    def build(self):
        """
        Método abstrato para construir a arquitetura do modelo.
        """
        pass

    @abstractmethod
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Método abstrato para treinar o modelo com os dados fornecidos.
        """
        pass

    @abstractmethod
    def evaluate(self, X_test, y_test):
        """
        Método abstrato para avaliar o modelo com os dados de teste fornecidos.
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Método abstrato para realizar predições usando o modelo treinado.
        """
        pass

    @abstractmethod
    def save(self, path):
        """
        Método abstrato para salvar o modelo treinado.
        """
        pass

    @abstractmethod
    def load(self, path):
        """
        Método abstrato para carregar um modelo treinado.
        """
        pass

class FOWMModel(ModeloDePredicao):
    def __init__(self, **config):
        super().__init__(**config)

    def build(self, params):
        """
        Configura o modelo FOWM com os parâmetros fornecidos.
        
        Args:
            params: Dicionário contendo os parâmetros do poço e do modelo.
        """
        self.model = FOWM(params)  # Inicializa o modelo FOWM
        self.trained = all(getattr(self.model, param) is not None for param in self.model.required_adjust_params)

    def train(self, X_train, y_train, n_particles=30, max_iter=100):
        """
        Ajusta os parâmetros do modelo FOWM usando PSO.
        """
        best_params, best_cost = self.model.fit(self.ssp0, X_train, y_train, n_particles, max_iter)
        self.trained = True
        self.history = {'best_params': best_params, 'best_cost': best_cost}
        return best_params, best_cost

    def evaluate(self, X_test, y_test):
        """
        Avalia o modelo FOWM com base nos dados de teste.
        """
        x_solution, _ = self.model.predict(*X_test)
        erro = np.mean((y_test - x_solution) ** 2)
        return erro

    def predict(self, X):
        """
        Realiza predições usando o modelo FOWM treinado.
        """
        if not self.trained:
            raise ValueError("O modelo deve ser treinado antes de realizar predições.")
        x_solution, _ = self.model.predict(*X)
        return x_solution

    def save(self, path):
        """
        Salva o estado atual do modelo FOWM.
        """
        # Salvar os parâmetros ajustados em um arquivo
        np.savez(path, **self.model.__dict__)

    def load(self, path):
        """
        Carrega um modelo FOWM treinado de um arquivo.
        """
        data = np.load(path, allow_pickle=True)
        for key in data:
            setattr(self.model, key, data[key].item())
        self.trained = True

class LSTMModel(ModeloDePredicao):
    def build(self):
        """
        Constrói a arquitetura do modelo LSTM.
        """
        # Configurações padrão ou especificadas pelo usuário
        activation = self.config.get('activation', 'tanh')
        optimizer = self.config.get('optimizer', 'adam')
        loss = self.config.get('loss', 'sparse_categorical_crossentropy')
        input_shape = self.config['input_shape']  # Defina input_shape de acordo com sua configuração

        # Construção da rede LSTM usando Keras Sequential
        self.model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=input_shape),  # Define a camada de entrada com a forma correta
            tf.keras.layers.LSTM(self.config.get('lstm_units', 50), return_sequences=True, activation=activation),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(self.config.get('lstm_units', 50), activation=activation),
            tf.keras.layers.Dense(self.config['num_classes'], activation='softmax')
        ])
        
        # Compilação do modelo com parâmetros especificados ou padrão
        self.model.compile(optimizer=optimizer, loss=loss, metrics=self.config.get('metrics', ['accuracy']))

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Treina o modelo LSTM com os dados de treinamento fornecidos.
        """
        self.history = self.model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=self.config['epochs'])
        self.trained = True

    def evaluate(self, X_test, y_test):
        """
        Avalia o modelo LSTM com os dados de teste fornecidos.
        """
        return self.model.evaluate(X_test, y_test)

    def predict(self, X):
        """
        Realiza predições usando o modelo LSTM treinado.
        """
        if not self.trained:
            raise ValueError("O modelo deve ser treinado antes de realizar predições.")
        return self.model.predict(X)

    def save(self, path):
        """
        Salva o modelo LSTM treinado no caminho especificado.
        """
        self.model.save(path)

    def load(self, path):
        """
        Carrega um modelo LSTM treinado do caminho especificado.
        """
        self.model = tf.keras.models.load_model(path)
        self.trained = True

class ANNModel(ModeloDePredicao):
    def build(self):
        """
        Constrói a arquitetura do modelo de rede neural artificial (ANN).
        """
        # Configurações padrão ou especificadas pelo usuário
        activation = self.config.get('activation', 'relu')
        optimizer = self.config.get('optimizer', 'adam')
        loss = self.config.get('loss', 'sparse_categorical_crossentropy')

        # Construção da rede neural usando Keras Sequential
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.InputLayer(input_shape=self.config['input_shape']))
        
        for units in self.config.get('hidden_layers', [64, 32]):
            self.model.add(tf.keras.layers.Dense(units, activation=activation))
        
        # Camada de saída
        self.model.add(tf.keras.layers.Dense(self.config['num_classes'], activation='softmax'))
        
        # Compilação do modelo com parâmetros especificados ou padrão
        self.model.compile(optimizer=optimizer, loss=loss, metrics=self.config.get('metrics', ['accuracy']))

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Treina o modelo ANN com os dados de treinamento fornecidos.
        
        Args:
            X_train: Conjunto de características de treinamento.
            y_train: Conjunto de rótulos de treinamento.
            X_val: (Opcional) Conjunto de características de validação.
            y_val: (Opcional) Conjunto de rótulos de validação.
        """
        # Treinamento do modelo
        self.history = self.model.fit(X_train, y_train, 
                                      validation_data=(X_val, y_val), 
                                      epochs=self.config.get('epochs', 50), 
                                      batch_size=self.config.get('batch_size', 32))
        self.trained = True

    def evaluate(self, X_test, y_test):
        """
        Avalia o modelo ANN com os dados de teste fornecidos.
        
        Args:
            X_test: Conjunto de características de teste.
            y_test: Conjunto de rótulos de teste.
            
        Returns:
            Métricas de avaliação do modelo.
        """
        return self.model.evaluate(X_test, y_test)

    def predict(self, X):
        """
        Realiza predições usando o modelo ANN treinado.
        
        Args:
            X: Conjunto de características para predição.
            
        Returns:
            Predições do modelo.
        """
        if not self.trained:
            raise ValueError("O modelo deve ser treinado antes de realizar predições.")
        return self.model.predict(X)

    def save(self, path):
        """
        Salva o modelo ANN treinado no caminho especificado.
        
        Args:
            path: Caminho para salvar o modelo.
        """
        self.model.save(path)

    def load(self, path):
        """
        Carrega um modelo ANN treinado do caminho especificado.
        
        Args:
            path: Caminho para carregar o modelo.
        """
        self.model = tf.keras.models.load_model(path)
        self.trained = True

class Controlador:
    def __init__(self, kp, ki, kd):
        """
        Inicializa o controlador PID.
        
        Args:
            kp: Ganho proporcional.
            ki: Ganho integral.
            kd: Ganho derivativo.
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0
        self.prev_error = 0

    def calcular_velocidade(self, setpoint, medida_atual):
        """
        Calcula a nova velocidade usando o controlador PID.
        
        Args:
            setpoint: Valor desejado.
            medida_atual: Valor atual medido.
            
        Returns:
            Novo valor de velocidade.
        """
        erro = setpoint - medida_atual
        self.integral += erro
        derivativo = erro - self.prev_error

        # Controlador PID
        velocidade = self.kp * erro + self.ki * self.integral + self.kd * derivativo

        # Atualiza o erro anterior
        self.prev_error = erro

        return velocidade
