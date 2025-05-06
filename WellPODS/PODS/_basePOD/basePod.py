import random

class Input:
    def __init__(self, value=None, config={'filter_function': None, 'noise_function': None}):
        """
        Classe para representar a entrada de dados de um POD.

        :param value: Valor inicial da entrada (pode ser constante ou dinâmico).
        :param config: Configurações adicionais para a entrada.
        :param filter_function: Função opcional para filtrar os dados.
        :param noise_function: Função opcional para adicionar ruído aos dados.
        """
        self.raw_value = value
        self.value = value
        self.config = config if config else {}
        self.filter_function = config.get('filter_function', None)
        self.noise_function = config.get('noise_function', None)

    def get(self):
        """
        Retorna o valor atual da entrada.
        """
        return self.value

    def set(self, value):
        """
        Define um novo valor para a entrada, com a possibilidade de filtrar ou adicionar ruído.

        :param value: Novo valor para a entrada.
        :param filter_function: Função opcional para filtrar os dados.
        :param noise_function: Função opcional para adicionar ruído aos dados.
        """
        self.value = value

        # Aplica a função de filtragem, se fornecida
        if self.filter_function:
            if isinstance(self.value, list):
                self.value = list(filter(self.filter_function, self.value))
            else:
                if not self.filter_function(self.value):
                    self.value = None

        # Aplica a função de ruído, se fornecida
        if self.noise_function and self.value is not None:
            if isinstance(self.value, list):
                self.value = [x + self.noise_function() for x in self.value]
            else:
                self.value += self.noise_function()
        
    def add_data(self, new_data):
        """
        Adiciona novos dados à entrada existente.

        :param new_data: Dados a serem adicionados (pode ser um único valor ou uma lista de valores).
        """
        if self.value is None:
            self.value = new_data
        elif isinstance(self.value, list):
            if isinstance(new_data, list):
                self.value.extend(new_data)
            else:
                self.value.append(new_data)
        else:
            self.value = [self.value, new_data]

    def filter_data(self, filter_function):
        """
        Filtra os dados da entrada usando uma função de primeira ordem.

        :param filter_function: Função que define o critério de filtragem.
        """
        if self.value is None:
            return
        if isinstance(self.value, list):
            self.value = list(filter(filter_function, self.value))
        else:
            if not filter_function(self.value):
                self.value = None

    def add_noise(self, noise_function):
        """
        Adiciona ruído aos dados da entrada usando uma função de ruído.

        :param noise_function: Função que gera o ruído a ser adicionado aos dados.
        """
        if self.value is None:
            return
        if isinstance(self.value, list):
            self.value = [x + noise_function() for x in self.value]
        else:
            self.value += noise_function()

class basePod:
    def __init__(self, name, model=None, config=None):
        """
        Classe genérica para representar um POD (ou módulo) no fluxo de simulação.

        :param name: Nome do POD.
        :param input_class: Classe a ser usada para criar a entrada (deve ser uma subclasse de Input).
        :param input_data: Dados de entrada para o POD.
        :param output_data: Dados de saída do POD.
        :param model: Modelo associado ao POD (função ou objeto que processa os dados).
        :param config: Parâmetros de configuração do POD (dicionário).
        """
        
        self.name = name
        self.model = model
        self.input_data = Input()
        self.output_data = None
        self.config = config if config else {}

    def process(self):
        """
        Processa os dados de entrada usando o modelo associado e gera os dados de saída.
        """
        if not self.model:
            raise ValueError(f"O POD '{self.name}' não possui um modelo definido.")
        if self.input_data is None or self.input_data.get() is None:
            raise ValueError(f"O POD '{self.name}' não possui dados de entrada.")
        
        # Processa os dados de entrada usando o modelo
        self.output_data = self.model(self.input_data.get(), **self.config)

    def set_input(self, input_data, config={'filter_function': None, 'noise_function': None}):
        """
        Define os dados de entrada para o POD.
        """
        if not isinstance(self.input_data, Input):
            raise ValueError(f"O POD '{self.name}' não possui uma classe de entrada definida.")
        if isinstance(input_data, list):
            self.input_data = Input(value=input_data, config=config)
        
        self.input_data.set(input_data)

    def get_output(self):
        """
        Retorna os dados de saída do POD.
        """
        return self.output_data

    def update_config(self, **kwargs):
        """
        Atualiza os parâmetros de configuração do POD.
        """
        self.config.update(kwargs)

# Exemplo de uso
if __name__ == "__main__":
    def exemplo_modelo(input_data, fator=1):
        return [x * fator for x in input_data]

    # Criando um POD
    pod = basePod(name="POD_Exemplo", model=exemplo_modelo, config={"fator": 2})

    # Definindo os dados de entrada
    pod.set_input([1, 2, 3, 4])
    print(f"Dados de entrada: {pod.input_data.get()}")  # Exibindo os dados de entrada
    
    # Processando os dados
    pod.process()

    # Obtendo os dados de saída
    print(f"Saída do POD: {pod.get_output()}")  # Modelo de exemplo que processa os dados

    # Exemplo com filtragem e ruído

    def filtro(x):
        return x > 2  # Filtra valores maiores que 2

    def ruido():
        return random.uniform(-0.5, 0.5)  # Adiciona ruído aleatório entre -0.5 e 0.5

    # Criando um Input com filtragem e ruído
    pod.set_input([1, 2, 3, 4, 5],
                  config={"filter_function": filtro})

    # Processando os dados
    pod.process()

    # Obtendo os dados de saída
    print(f"Saída do POD com filtro: {pod.get_output()}")  # Modelo de exemplo que processa os dados

    pod.set_input([1, 2, 3, 4, 5],
                  config={"noise_function": ruido})

    # Processando os dados
    pod.process()

    # Obtendo os dados de saída
    print(f"Saída do POD com ruído: {pod.get_output()}")  # Modelo de exemplo que processa os dados
