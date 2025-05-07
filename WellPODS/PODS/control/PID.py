class PID:
    """
    Classe para implementar um controlador PID para uma válvula choke.
    """
    def __init__(self, kp, ki, kd, type='ISA'):
        """
        Inicializa o controlador PID.
        
        Args:
            kp: Ganho proporcional.
            ki: Ganho integral.
            kd: Ganho derivativo.
            type: Tipo de controlador (ex.: 'ISA').
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0
        self.prev_error = 0
        self.type = type

    def set_params(self, kp=None, ki=None, kd=None, type=None):
        """
        Define os parâmetros do controlador PID.
        
        Args:
        kp: Novo valor para o ganho proporcional (opcional).
        ki: Novo valor para o ganho integral (opcional).
        kd: Novo valor para o ganho derivativo (opcional).
        type: Novo tipo de controlador (opcional).
        """
        if kp is not None:
            self.kp = kp
        if ki is not None:
            self.ki = ki
        if kd is not None:
            self.kd = kd
        if type is not None:
            self.type = type
                
    def calcular_acao_controle(self, setpoint, medida_atual):
        """
        Calcula o sinal de controle para a válvula choke.
        
        Args:
            setpoint: Valor desejado (setpoint).
            medida_atual: Valor atual medido.
            
        Returns:
            Sinal de controle para a válvula choke.
        """
        erro = setpoint - medida_atual
        self.integral += erro
        derivativo = erro - self.prev_error

        # Controlador PID
        sinal_controle = self.kp * erro + self.ki * self.integral + self.kd * derivativo

        # Atualiza o erro anterior
        self.prev_error = erro

        return sinal_controle