class ChokeValve:
    """
    Classe que representa a válvula choke controlada pelo PID.
    """
    def __init__(self, abertura_inicial=0):
        """
        Inicializa a válvula choke.
        
        Args:
            abertura_inicial: Posição inicial da válvula (0 a 100%).
        """
        self.abertura = abertura_inicial

    def aplicar_sinal(self, sinal):
        """
        Aplica o sinal de controle à válvula choke.
        
        Args:
            sinal: Sinal de controle calculado pelo PID.
        """
        # Atualiza a abertura da válvula com base no sinal de controle
        self.abertura += sinal

        # Garante que a abertura esteja entre 0% e 100%
        self.abertura = max(0, min(100, self.abertura))

    def obter_abertura(self):
        """
        Retorna a abertura atual da válvula.
        
        Returns:
            Abertura da válvula (0 a 100%).
        """
        return self.abertura