from ..utils.Input import Input  # Caminho relativo para importar a classe Input

class Process(Input):
    """
    Classe para representar a entrada do processo.
    """
    def __init__(self, valor_inicial):
        super().__init__(valor_inicial)
