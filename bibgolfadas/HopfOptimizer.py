import numpy as np

class StochasticOptimizer:
    def __init__(self, learning_rate=0.01, max_iter=100, batch_size=32, tol=1e-6, random_seed=None):
        """
        Inicializa o otimizador estocástico com os hiperparâmetros necessários.

        :param learning_rate: Taxa de aprendizado.
        :param max_iter: Número máximo de iterações.
        :param batch_size: Tamanho do mini-batch.
        :param tol: Tolerância para critério de parada.
        :param random_seed: Semente para a aleatoriedade (reprodutibilidade).
        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.tol = tol
        self.random_seed = random_seed
        self.history = []

        if random_seed is not None:
            np.random.seed(random_seed)

    def objective(self, params, batch):
        """
        Função objetivo, ajustada para trabalhar com mini-batches.

        :param params: Parâmetros a serem otimizados.
        :param batch: Mini-batch dos dados.
        :return: Valor da função objetivo para o mini-batch.
        """

        
        raise NotImplementedError("Função objetivo não implementada!")

    def gradient(self, params, batch):
        """
        Calcula o gradiente para o mini-batch de dados.

        :param params: Parâmetros atuais.
        :param batch: Mini-batch dos dados.
        :return: Gradiente calculado para o mini-batch.
        """
        raise NotImplementedError("Cálculo do gradiente não implementado!")

    def optimize(self, initial_params, data):
        """
        Realiza o processo de otimização.

        :param initial_params: Parâmetros iniciais para a otimização.
        :param data: Conjunto de dados completo.
        :return: Parâmetros otimizados.
        """
        params = initial_params
        n_samples = len(data)

        for i in range(self.max_iter):
            # Seleciona um mini-batch aleatório
            batch_indices = np.random.choice(n_samples, self.batch_size, replace=False)
            batch = data[batch_indices]

            # Calcula o gradiente para o mini-batch
            grad = self.gradient(params, batch)
            params = self.update(params, grad)

            # Salva o histórico da função objetivo (opcional)
            self.history.append(self.objective(params, batch))

            # Critério de parada
            if abs(self.history[-1]) < self.tol:
                print(f"Convergência alcançada em {i + 1} iterações.")
                break

        return params

    def update(self, params, grad):
        """
        Atualiza os parâmetros com base no gradiente estocástico.

        :param params: Parâmetros atuais.
        :param grad: Gradiente do mini-batch.
        :return: Novos parâmetros atualizados.
        """
        return params - self.learning_rate * grad
