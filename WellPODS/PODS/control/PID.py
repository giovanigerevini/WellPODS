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

    def calcular_sinal(self, setpoint, medida_atual):
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