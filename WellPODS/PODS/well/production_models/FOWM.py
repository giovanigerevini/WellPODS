import numpy as np
from functools import partial
from scipy.integrate import odeint # Import odeint
from PyDSTool import *

class FOWM:
    """
    Classe do modelo FOWM

    2017 - Diehl, Fabio C. - Fast Offshore Wells Model (FOWM): A practical dynamic model for multiphase oil production systems in deepwater and ultra-deepwater scenarios - Computers & Chemical Engineering.

    disponível em: https://www.sciencedirect.com/science/article/pii/S0098135417300443

    """
    def __init__(self, params):
        """
        Inicializa a classe FOWM com os parâmetros fornecidos.
        
        Args:
            params: Dicionário contendo os parâmetros específicos do poço.
        """
        # Definindo os atributos diretamente a partir dos parâmetros
        self.ssp0 = [1000, 1000, 1000, 1000, 1000, 1000]  # Estado inicial padrão

        for key, value in params.items():
            setattr(self, key, value)

        # Parâmetros de ajuste obrigatórios
        self.required_adjust_params = ['mlstill', 'Cg', 'Cout', 'Veb', 'Epsilon', 'Kw', 'Ka', 'Kr', 'Vr']
        
        # Inicializa os parâmetros de ajuste como None se não forem fornecidos
        for param in self.required_adjust_params:
            if not hasattr(self, param):
                setattr(self, param, None)

        # Atributo para indicar se o modelo foi treinado (parâmetros definidos)
        self.trained = False

        # Verifica se todos os parâmetros de ajuste foram fornecidos e define o estado de treinamento
        self.check_required_adjust_params()

        # Calcula as constantes e atributos derivados
        self.A = np.pi * self.D ** 2 / 4
        self.Vt = self.Lt * np.pi * self.Dt ** 2 / 4
        self.Va = self.La * np.pi * self.Da ** 2 / 4
        self.Vr = self.L * np.pi * self.D ** 2 / 4
        self.fat_Wgc = 101325 * self.M / (293 * self.R) / 3600 / 24
        self.RT = self.R * self.T
        self.RT_over_M = self.R * self.T / self.M
        self.g_sin_teta_over_A = self.g * np.sin(self.teta) / self.A
        self.g_Hvgl = self.g * self.Hvgl
        self.g_Hvgl_over_Vt = self.g * self.Hvgl / self.Vt
        self.Romres_g_deltaGlPdg = self.Romres * self.g * (self.Hpdg - self.Hvgl)
        self.Romres_g_deltaWellPdg = self.Romres * self.g * (self.Ht - self.Hpdg)
        self.anular_pressure = self.R * self.T / self.M / self.Va + self.g * self.La / self.Va
        
    def check_required_adjust_params(self):
        """
        Verifica se todos os parâmetros de ajuste obrigatórios foram definidos.
        Se todos estiverem definidos, define o atributo 'trained' como True.
        Caso contrário, define 'trained' como False e gera um erro.
        """
        undefined_params = [param for param in self.required_adjust_params if getattr(self, param) is None]
        if undefined_params:
            self.trained = False
            raise ValueError(f"Parâmetros de ajuste não definidos: {', '.join(undefined_params)}.")
        else:
            self.trained = True

    def set_params(self, **kwargs):
        """
        Define ou atualiza os parâmetros de ajuste do modelo.
        
        Args:
            kwargs: Dicionário com os parâmetros de ajuste.
        """
        for param, value in kwargs.items():
            if param in self.required_adjust_params:
                setattr(self, param, value)
        self.check_required_adjust_params()  # Verifica e atualiza o estado de 'trained' após definir os parâmetros

    def predict_y(self, t, ssp, z, Wgc, Pr, Ps):
        """
        """
        all_vars = {
            'z': [],
            'Wgc': [],
            'Pr': [],
            'Ps': [],
            'x1t': [],
            'x2t': [],
            'x3t': [],
            'x4t': [],
            'x5t': [],
            'x6t': [],
            'Peb': [],
            'Prt': [],
            'Prb': [],
            'ALFAg': [],
            'Wout': [],
            'Wg': [],
            'Vgt': [],
            'ROgt': [],
            'ROmt': [],
            'Ptt': [],
            'Ptb': [],
            'Ppdg': [],
            'Pbh': [],
            'ALFAgt': [],
            'Wwh': [],
            'Wr': [],
            'Pai': [],
            'ROai': [],
            'Wiv': [],
            'Wlout': [],
            'Wgout': [],
            't': t,
        }

        # Preenchendo as variáveis iniciais
        all_vars['z'] = np.array([z(ti) for ti in t]) if callable(z) else np.ones(t.shape[0]) * z
        all_vars['Wgc'] = np.array([Wgc(ti) for ti in t]) if callable(Wgc) else np.ones(t.shape[0]) * Wgc
        all_vars['Pr'] = np.array([Pr(ti) for ti in t]) if callable(Pr) else np.ones(t.shape[0]) * Pr
        all_vars['Ps'] = np.array([Ps(ti) for ti in t]) if callable(Ps) else np.ones(t.shape[0]) * Ps

        try:
            all_vars['x1t'], all_vars['x2t'], all_vars['x3t'], all_vars['x4t'], all_vars['x5t'], all_vars['x6t'] = (
                ssp[:, 0], ssp[:, 1], ssp[:, 2], ssp[:, 3], ssp[:, 4], ssp[:, 5]
            )
        except:
            all_vars['x1t'], all_vars['x2t'], all_vars['x3t'], all_vars['x4t'], all_vars['x5t'], all_vars['x6t'] = (
                [ssp[0]] * t.shape[0], [ssp[1]] * t.shape[0], [ssp[2]] * t.shape[0], [ssp[3]] * t.shape[0], [ssp[4]] * t.shape[0], [ssp[5]] * t.shape[0]
            )

        # Calculando as variáveis internas e armazenando nos dicionários
        all_vars['Peb'] = all_vars['x1t'] * self.RT_over_M / self.Veb
        all_vars['Prt'] = all_vars['x2t'] * self.RT_over_M / (self.Vr - (all_vars['x3t'] + self.mlstill) / self.Rol)
        all_vars['Prb'] = all_vars['Prt'] + (all_vars['x3t'] + self.mlstill) * self.g_sin_teta_over_A
        all_vars['ALFAg'] = all_vars['x2t'] / (all_vars['x2t'] + all_vars['x3t'])
        all_vars['Wout'] = [
            self.Cout * all_vars['z'][i] * np.sqrt(self.Rol * max(0, (all_vars['Prt'][i] - all_vars['Ps'][i])))
            for i in range(len(all_vars['z']))
        ]
        all_vars['Wg'] = self.Cg * np.maximum(0, (all_vars['Peb'] - all_vars['Prb']))
        all_vars['Vgt'] = self.Vt - all_vars['x6t'] / self.Rol
        all_vars['ROgt'] = all_vars['x5t'] / all_vars['Vgt']
        all_vars['ROmt'] = (all_vars['x5t'] + all_vars['x6t']) / self.Vt
        all_vars['Ptt'] = all_vars['ROgt'] * self.RT_over_M
        all_vars['Ptb'] = all_vars['Ptt'] + all_vars['ROmt'] * self.g * self.Hvgl
        all_vars['Ppdg'] = all_vars['Ptb'] + self.Romres_g_deltaGlPdg
        all_vars['Pbh'] = all_vars['Ppdg'] + self.Romres_g_deltaWellPdg
        all_vars['ALFAgt'] = all_vars['x5t'] / (all_vars['x5t'] + all_vars['x6t'])
        all_vars['Wwh'] = self.Kw * np.sqrt(self.Rol * np.maximum(0, (all_vars['Ptt'] - all_vars['Prb'])))
        all_vars['Wr'] = self.Kr * (1 - 0.2 * all_vars['Pbh'] / all_vars['Pr'] - 0.8 * (all_vars['Pbh'] / all_vars['Pr']) ** 2)
        all_vars['Pai'] = self.anular_pressure * all_vars['x4t']
        all_vars['ROai'] = all_vars['Pai'] / self.RT_over_M
        all_vars['Wiv'] = self.Ka * np.sqrt(np.maximum(0, (all_vars['Pai'] - all_vars['Ptb'])))
        all_vars['Wlout'] = (1 - all_vars['ALFAg']) * all_vars['Wout']
        all_vars['Wgout'] = all_vars['ALFAg'] * all_vars['Wout']
        
        return all_vars

    def predict(self, X, internal_vars_requested=None):
        """
        Predict the system's response using the optimized parameters and initial conditions.
        
        Args:
            ssp0: Initial state space vector
            tArray_sim: Time array for simulation
            z_sim: Choke input array
            Wgc_sim: Gas lift flow input array
            Pr_sim: Reservoir pressure array
            Ps_sim: Production header pressure array
            internal_vars_requested: List of internal variable names to return

        Returns:
            x_solution: Integrated solution over the time array
            internal_vars: Dictionary containing only the requested internal variables over time
        """
        # Verifica se todos os parâmetros de ajuste foram definidos
        self.check_required_adjust_params()

        tArray_sim, z_sim, Wgc_sim, Pr_sim, Ps_sim = X

        def get_input_value(input, t):
            return input(t) if callable(input) else input

        if internal_vars_requested is None:
            internal_vars_requested = []

        def Velocity_disturb_collect(ssp, t, z_input, Wgc_input, Pr_input, Ps_input):
            # Read state space points
            x1, x2, x3, x4, x5, x6 = ssp  
            
            # Obter o valor de cada entrada no tempo t
            z = get_input_value(z_input, t)
            Wgc = get_input_value(Wgc_input, t)
            Pr = get_input_value(Pr_input, t)
            Ps = get_input_value(Ps_input, t)
            
            # Cálculos das variáveis internas (como no Velocity_disturb)
            Peb   = x1 * self.RT_over_M / self.Veb
            Prt   = x2 * self.RT_over_M / (self.Vr - (x3 + self.mlstill) / self.Rol)
            Prb   = Prt + (x3 + self.mlstill) * self.g_sin_teta_over_A
            ALFAg = x2 / (x2 + x3)
            Wout  = self.Cout * z * np.sqrt((self.Rol * max(0, (Prt - Ps))))
            Wg    = self.Cg * max(0, (Peb - Prb))
            Vgt   = self.Vt - x6 / self.Rol
            ROgt  = x5 / Vgt
            ROmt  = (x5 + x6) / self.Vt
            Ptt   = ROgt * self.RT_over_M
            Ptb   = Ptt + ROmt * self.g_Hvgl
            Ppdg  = Ptb + self.Romres_g_deltaGlPdg
            Pbh   = Ppdg + self.Romres_g_deltaWellPdg
            ALFAgt = x5 / (x5 + x6)
            Wwh   = self.Kw * np.sqrt(self.Rol * max(0, (Ptt - Prb)))
            Wr_pre = self.Kr * (1 - 0.2 * Pbh / Pr - 0.8 * (Pbh / Pr) ** 2)
            Wr    = max(0, Wr_pre)
            Pai  = self.anular_pressure * x4
            ROai = Pai / self.RT_over_M
            Wiv  = self.Ka * np.sqrt(ROai * max(0, (Pai - Ptb)))
            
            # Retorno dos valores do sistema dinâmico
            dx1 = (1 - self.Epsilon) * ALFAgt * Wwh - Wg
            dx2 = self.Epsilon * Wwh * ALFAgt + Wg - ALFAg * Wout
            dx3 = Wwh * (1 - ALFAgt) - (1 - ALFAg) * Wout
            dx4 = Wgc - Wiv
            dx5 = Wr * self.ALFAgw + Wiv - Wwh * ALFAgt
            dx6 = Wr * (1 - self.ALFAgw) - Wwh * (1 - ALFAgt) 
            
            vel = np.array([dx1, dx2, dx3, dx4, dx5, dx6], float)
            return vel

        # Solução da EDO com coleta de variáveis internas
        x_solution = odeint(
            Velocity_disturb_collect,
            self.ssp0, 
            tArray_sim, 
            args=(z_sim, Wgc_sim, Pr_sim, Ps_sim),
            rtol=1e-9,
            atol=1e-11
        )
 
        if internal_vars_requested:
            all_internal_vars = self.predict_y(
                tArray_sim,
                x_solution,
                z_sim,
                Wgc_sim,
                Pr_sim,
                Ps_sim
            )

        # Filtrar apenas as variáveis internas solicitadas
        internal_vars = {var: all_internal_vars[var] for var in internal_vars_requested if var in all_internal_vars}

        return x_solution, internal_vars

    def build_bifurcation(self, input = 'Ck', out = 'Ppdg', range_values = [2,100], internal_vars_requested=None):
        """
        Realiza uma análise de bifurcação do modelo FOWM para identificar mudanças qualitativas
        no comportamento dinâmico do sistema em função do parâmetro de controle.

        Args:
            input (str): Nome da entrada para bifurcação ('Ck', 'GL', 'Ps', ou 'Pr').
            range_values (list): Intervalo de valores para o parâmetro de entrada.

        Returns:
            ssp_cont: Soluções estacionárias do modelo durante a bifurcação.
            bifurcation_curve: Dados da curva de bifurcação.
            fowm_Hopf: Dados relacionados ao ponto de Hopf bifurcação, se aplicável.
            hopf_params: Parâmetros associados aos pontos de Hopf.
        """
        # Verifica se todos os parâmetros de ajuste foram definidos
        self.check_required_adjust_params()

        # Definir os nomes e valores dos parâmetros do modelo
        parameters_name = [
            'R', 'g', 'T', 'M', 'Rol', 'ALFAgw', 'Romres',
            'L', 'Lt', 'La', 'D', 'Dt', 'Da', 'teta', 'Hvgl', 'Hpdg', 'Ht', 'A', 'Vt', 'Va',
            'RT_over_M', 'fat_Wgc', 'g_sin_teta_over_A',
            'Romres_g_deltaGlPdg', 'Romres_g_deltaWellPdg', 'anular_pressure',
            'mlstill', 'Cg', 'Cout', 'Veb', 'Epsilon', 'Kw', 'Ka', 'Kr', 'Vr',
            'Ck', 'GL', 'Ps', 'Pr'
        ]
        
        parameters = [
            self.R, self.g, self.T, self.M, self.Rol, self.ALFAgw, self.Romres,
            self.L, self.Lt, self.La, self.D, self.Dt, self.Da, self.teta, self.Hvgl, self.Hpdg, self.Ht, self.A, self.Vt, self.Va,
            self.RT_over_M, self.fat_Wgc, self.g_sin_teta_over_A,
            self.Romres_g_deltaGlPdg, self.Romres_g_deltaWellPdg, self.anular_pressure,
            self.mlstill, self.Cg, self.Cout, self.Veb, self.Epsilon, self.Kw, self.Ka, self.Kr, self.Vr,
            self.Ck0, self.GL0, self.Ps0, self.Pr0
        ]

        parameters_dict = dict(zip(parameters_name, parameters))

        icdict = {
            'x1': self.ssp0[0], 'x2': self.ssp0[1], 'x3': self.ssp0[2],
            'x4': self.ssp0[3], 'x5': self.ssp0[4], 'x6': self.ssp0[5]
        }

        # Definir as especificações do modelo
        symbolFOWM = {
            'z'     : (['x1','x2','x3','x4','x5','x6'],'Ck*0.01'), \
            'Wgc'   : (['x1','x2','x3','x4','x5','x6'],'GL*fat_Wgc'), \
            'Peb'   : (['x1','x2','x3','x4','x5','x6'],'x1*RT_over_M/Veb'), \
            'Prt'   : (['x1','x2','x3','x4','x5','x6'],'x2*RT_over_M/(Vr-(x3+mlstill)/Rol)'), \
            'Prb'   : (['x1','x2','x3','x4','x5','x6'],'Prt(x1,x2,x3,x4,x5,x6) + (x3 + mlstill)*g_sin_teta_over_A'), \
            'ALFAg' : (['x1','x2','x3','x4','x5','x6'],'x2/(x2 + x3)'), \
            'Wout'  : (['x1','x2','x3','x4','x5','x6'],'Cout*z(x1,x2,x3,x4,x5,x6)*sqrt((Rol*(max(0,Prt(x1,x2,x3,x4,x5,x6) - Ps))))'), \
            'Wlout' : (['x1','x2','x3','x4','x5','x6'],'(1 - ALFAg(x1,x2,x3,x4,x5,x6))*Wout(x1,x2,x3,x4,x5,x6)'), \
            'Wgout' : (['x1','x2','x3','x4','x5','x6'],'ALFAg(x1,x2,x3,x4,x5,x6)*Wout(x1,x2,x3,x4,x5,x6)'), \
            'Wg'    : (['x1','x2','x3','x4','x5','x6'],'Cg*(max(0,Peb(x1,x2,x3,x4,x5,x6) - Prb(x1,x2,x3,x4,x5,x6)))'), \
            'Vgt'   : (['x1','x2','x3','x4','x5','x6'],'Vt - x6/Rol'),  \
            'ROgt'  : (['x1','x2','x3','x4','x5','x6'],'x5/Vgt(x1,x2,x3,x4,x5,x6)'),     \
            'ROmt'  : (['x1','x2','x3','x4','x5','x6'],'(x5 + x6)/Vt'),  \
            'Ptt'   : (['x1','x2','x3','x4','x5','x6'],'ROgt(x1,x2,x3,x4,x5,x6)*RT_over_M'),		\
            'Ptb'   : (['x1','x2','x3','x4','x5','x6'],'Ptt(x1,x2,x3,x4,x5,x6) + ROmt(x1,x2,x3,x4,x5,x6)*g*Hvgl'),	\
            'Ppdg'  : (['x1','x2','x3','x4','x5','x6'],'Ptb(x1,x2,x3,x4,x5,x6) + Romres_g_deltaGlPdg'),	\
            'Pbh'   : (['x1','x2','x3','x4','x5','x6'],'Ppdg(x1,x2,x3,x4,x5,x6) + Romres_g_deltaWellPdg') ,   \
            'ALFAgt': (['x1','x2','x3','x4','x5','x6'],'x5/(x6 + x5)')  ,      \
            'Wwh'   : (['x1','x2','x3','x4','x5','x6'],'Kw*sqrt(Rol*(max(0,Ptt(x1,x2,x3,x4,x5,x6) - Prb(x1,x2,x3,x4,x5,x6))))'),   \
            'Wwhg'  : (['x1','x2','x3','x4','x5','x6'],'Wwh(x1,x2,x3,x4,x5,x6)*ALFAgt(x1,x2,x3,x4,x5,x6)'),\
            'Wwhl'  : (['x1','x2','x3','x4','x5','x6'],'Wwh(x1,x2,x3,x4,x5,x6)*(1 - ALFAgt(x1,x2,x3,x4,x5,x6))'),	\
            'Wr_pre': (['x1','x2','x3','x4','x5','x6'],'Kr*(1-0.2*Pbh(x1,x2,x3,x4,x5,x6)/Pr-0.8*(Pbh(x1,x2,x3,x4,x5,x6)/Pr)**2)'),		\
            'Wr'    : (['x1','x2','x3','x4','x5','x6'],'max(0,Wr_pre(x1,x2,x3,x4,x5,x6))'),			\
            'Pai'   : (['x1','x2','x3','x4','x5','x6'],'anular_pressure*x4'),     \
            'ROai'  : (['x1','x2','x3','x4','x5','x6'],'Pai(x1,x2,x3,x4,x5,x6)/RT_over_M') ,                       \
            'Wiv'   : (['x1','x2','x3','x4','x5','x6'],'Ka*sqrt(ROai(x1,x2,x3,x4,x5,x6)*(max(0,Pai(x1,x2,x3,x4,x5,x6) - Ptb(x1,x2,x3,x4,x5,x6))))')
        }

        dx = {
            'x1': '(1 - Epsilon)*(Wwhg(x1,x2,x3,x4,x5,x6)) - Wg(x1,x2,x3,x4,x5,x6)',
            'x2': 'Epsilon*(Wwhg(x1,x2,x3,x4,x5,x6)) + Wg(x1,x2,x3,x4,x5,x6) - Wgout(x1,x2,x3,x4,x5,x6)',
            'x3': 'Wwhl(x1,x2,x3,x4,x5,x6) - Wlout(x1,x2,x3,x4,x5,x6)',
            'x4': 'Wgc(x1,x2,x3,x4,x5,x6) - Wiv(x1,x2,x3,x4,x5,x6)',
            'x5': 'Wr(x1,x2,x3,x4,x5,x6) * ALFAgw + Wiv(x1,x2,x3,x4,x5,x6) - Wwhg(x1,x2,x3,x4,x5,x6)',
            'x6': 'Wr(x1,x2,x3,x4,x5,x6) * (1 - ALFAgw) - Wwhl(x1,x2,x3,x4,x5,x6)'
        }

        DSargs = args(name='FOWM')
        DSargs.pars = parameters_dict
        DSargs.varspecs = dx
        DSargs.fnspecs = symbolFOWM
        DSargs.ics = icdict
        DSargs.pdomain = {input: range_values}
        DSargs.xdomain = {input: range_values}

        testDS = Generator.Vode_ODEsystem(DSargs)

        # Configuração para o cálculo da bifurcação
        PCargs = args(name='EQ1', type='EP-C')
        PCargs.freepars = [input]
        PCargs.StepSize = 1e-1
        PCargs.MaxNumPoints = 400
        PCargs.MaxStepSize = 1e1
        PCargs.LocBifPoints = 'all'
        PCargs.StopAtPoints = 'B'
        PCargs.SaveEigen = True
        PCargs.verbosity = 2

        PyCont = ContClass(testDS)
        PyCont.newCurve(PCargs)

        print('Computing bifurcation curve...')
        start = perf_counter()
        PyCont['EQ1'].forward()
        PyCont['EQ1'].backward()
        PyCont['EQ1'].backward()
        PyCont['EQ1'].backward()
        elapsed_time = perf_counter() - start
        print(f'Done in {elapsed_time:.3f} seconds!')

        # Extrair soluções da bifurcação
        ssp_cont = np.array([
            PyCont['EQ1'].sol['x1'],
            PyCont['EQ1'].sol['x2'],
            PyCont['EQ1'].sol['x3'],
            PyCont['EQ1'].sol['x4'],
            PyCont['EQ1'].sol['x5'],
            PyCont['EQ1'].sol['x6']
        ]).T

        # Função base para criar a bifurcação
        def configure_bifurcation_curve(sol, ck0, gl0, pr0, ps0, internal_vars_requested=0, extra=None):
            if internal_vars_requested:
                all_internal_vars = self.predict_y(
                    tArray_sim,
                    x_solution,
                    z_sim,
                    Wgc_sim,
                    Pr_sim,
                    Ps_sim
                )

            # Filtrar apenas as variáveis internas solicitadas
            internal_vars = {var: all_internal_vars[var] for var in internal_vars_requested if var in all_internal_vars}
            # Adiciona o extra à bifurcation_curve se aplicável
            curve = self.predict_y(
                np.arange(0, ssp_cont.shape[0]),
                ssp_cont,
                ck0,
                gl0,
                pr0,
                ps0,
            )
            if extra:
                curve[extra] = sol[extra]
            return curve

        # Mapeamento de entradas para configurações dinâmicas
        configure_map = {
            'Ck': lambda sol: configure_bifurcation_curve(sol, sol['Ck'] * 0.01, self.GL0 * self.fat_Wgc, self.Pr0, self.Ps0),
            'GL': lambda sol: configure_bifurcation_curve(sol, self.Ck0, sol['GL'] * self.fat_Wgc, self.Pr0, self.Ps0),
            'Ps': lambda sol: configure_bifurcation_curve(sol, self.Ck0, self.GL0 * self.fat_Wgc, self.Pr0, sol['Ps']),
            'Pr': lambda sol: configure_bifurcation_curve(sol, self.Ck0, self.GL0 * self.fat_Wgc, sol['Pr'], self.Ps0)
        }

        # Verifica se o input é um dos padrões ou um parâmetro válido no modelo
        if input in configure_map:
            bifurcation_curve = configure_map[input](PyCont['EQ1'].sol)
        elif input in parameters_name:
            # Caso o input seja um parâmetro válido no modelo, adicionar uma nova coluna
            bifurcation_curve = configure_bifurcation_curve(
                PyCont['EQ1'].sol,
                self.Ck0,
                self.GL0 * self.fat_Wgc,
                self.Pr0,
                self.Ps0,
                extra=input
            )
        else:
            raise ValueError(f"Parâmetro de entrada '{input}' não reconhecido ou inválido.")

        choke_Hopff = PyCont['EQ1'].getSpecialPoint('H1')[input] if input in PyCont['EQ1'].getSpecialPoint('H1') else None
        fowm_Hopf = bifurcation_curve[bifurcation_curve.Choke == choke_Hopff] if choke_Hopff else None

        return ssp_cont, bifurcation_curve, fowm_Hopf