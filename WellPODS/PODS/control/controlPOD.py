# from ._basePOD.basePOD import basePOD
from WellPODS.PODS._basePOD.basePOD import basePOD
from WellPODS.PODS.control.PID import PID

class ControlPOD(basePOD):
    def __init__(self, id, control_model, POD_type='control'):
        if control_model == 'PID':
            super().__init__(id, POD_type=POD_type, 
                             model=PID(kp=None, ki=None, kd=None, type=None))

    def set_params(self, **kwargs):
        if hasattr(self.model, 'set_params'):
            self.model.set_params(**kwargs)
        else:
            raise AttributeError("The model does not have a 'set_params' method.")
    
    def calcular_acao_controle(self, setpoint, medida_atual):
        # [TODO]
        pass
    
    def get_model(self):
        return self.model

# Exemplo de uso da classe ControlPOD
if __name__ == "__main__":
    # Criando um modelo de exemplo
    modelo_exemplo = 'PID'

    # Instanciando a classe ControlPOD
    pid_control = ControlPOD('1', modelo_exemplo)
    
    # inicializando o controlador PID com parâmetros
    pid_control.set_params(kp=1.0, ki=0.1, kd=0.01, type='ISA')
    
    # Definindo os dados de entrada
    pid_control.set_input([10, 20, 30, 40])
    
    print(f"Dados de entrada: {pid_control.input_data.get()}")  # Exibindo os dados de entrada
    
    # Processando os dados
    pid_control.process()
    
    print(f"Saída do POD: {pid_control.get_output()}")  # Modelo de exemplo que processa os dados
    
    