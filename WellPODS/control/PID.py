class PID:
    """
    Descrevre a classe
    """
    def __init__(self, kp, ki, kd, type='ISA'):
        """
        Inicializa o controlador PID.
        
        Args:
            kp: Ganho proporcional.
            ki: Ganho integral.
            kd: Ganho derivativo.
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0
        self.prev_error = 0
        self.type = type

    def calcular_velocidade(self, setpoint, medida_atual):
        """
        Calcula a nova velocidade usando o controlador PID.
        
        Args:
            setpoint: Valor desejado.
            medida_atual: Valor atual medido.
            
        Returns:
            Novo valor de velocidade.
        """
        erro = setpoint - medida_atual
        self.integral += erro
        derivativo = erro - self.prev_error

        # Controlador PID
        velocidade = self.kp * erro + self.ki * self.integral + self.kd * derivativo

        # Atualiza o erro anterior
        self.prev_error = erro

        return velocidade
