from ..utils.Input import Input  # Caminho relativo para importar a classe Input

class GasLiftSupply(Input):
    """
    Classe para representar a entrada de fornecimento de gas-lift.
    """
    def __init__(self, valor_inicial):
        super().__init__(valor_inicial)
