from abc import ABC, abstractmethod

class Otimizador(ABC):
    def __init__(self):
        """
        Classe base abstrata para otimizadores. Define a estrutura básica para criação de otimizadores específicos.
        Inclui um log para registrar eventos durante a otimização.
        """
        self.method = None
        self.objective_function = None
        self.params = None
        self.log = []  # Lista para armazenar mensagens de log

    @abstractmethod
    def build(self, method, objective_function, initial_params):
        """
        Método para configurar o otimizador com um método de otimização específico, função objetivo e parâmetros iniciais.

        :param method: Método de otimização a ser usado (ex: 'sgd', 'adam', 'scipy', etc.).
        :param objective_function: Função objetivo a ser minimizada ou maximizada.
        :param initial_params: Parâmetros iniciais para a otimização.
        """
        self.method = method
        self.objective_function = objective_function
        self.params = initial_params
        self.add_log(f"Otimizador configurado com método: {method}")

    @abstractmethod
    def optimize(self):
        """
        Método para executar a otimização com base na configuração fornecida pelo método build.
        Deve ser implementado pelas subclasses.
        """
        pass

    def add_log(self, message):
        """
        Adiciona uma mensagem ao log.

        :param message: Mensagem de log a ser registrada.
        """
        self.log.append(message)

    def show_log(self):
        """
        Mostra o log registrado até o momento.
        """
        print("\nLog de Otimização:")
        for entry in self.log:
            print(entry)