import bibgolfadas.FOWM as FOWM
import numpy as np
import tensorflow as tf

from abc import ABC, abstractmethod

class ModeloDePredicao(ABC):
    """
    Representa ode modelos de predição disponíveis

    Atributos:
        make (str): Marca do carro.
        model (str): Modelo do carro.
        year (int): Ano de fabricação do carro.
        speed (float): Velocidade atual do carro em km/h.
    """

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
        self.model = FOWM.FOWM(params)  # Inicializa o modelo FOWM
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
