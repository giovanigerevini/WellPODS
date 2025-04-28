from ..utils.Input import Input  # Caminho relativo para importar a classe Input

class Reservoir(Input):
    """
    Classe para representar a entrada do reservatório.
    """
    def __init__(self, valor_inicial):
        super().__init__(valor_inicial)
