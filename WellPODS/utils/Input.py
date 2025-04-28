class Input:
    """
    Classe genérica para representar uma entrada de dados com valor inicial obrigatório.
    """
    def __init__(self, valor_inicial):
        """
        Inicializa a entrada com um valor inicial.
        
        Args:
            valor_inicial: O valor inicial obrigatório da entrada.
        """
        if valor_inicial is None:
            raise ValueError("O 'valor_inicial' é obrigatório.")
        self.valor_inicial = valor_inicial
        self.valor_atual = valor_inicial

    def atualizar_valor(self, novo_valor):
        """
        Atualiza o valor atual da entrada.
        
        Args:
            novo_valor: O novo valor a ser definido.
        """
        self.valor_atual = novo_valor

    def ler_valor(self):
        """
        Retorna o valor atual da entrada.
        
        Returns:
            O valor atual.
        """
        return self.valor_atual

    def reiniciar_valor(self):
        """
        Reinicia o valor atual para o valor inicial.
        """
        self.valor_atual = self.valor_inicial