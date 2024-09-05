# Importando as classes necessárias
import matplotlib.pyplot as plt

import bibgolfadas.bibgolfadas as bibgolfadas
from bibgolfadas.Tools import transformar_entrada, normalizar_zscore, generate_dynamic_input
import numpy as np
import pandas as pd

from importlib import reload
bibgolfadas = reload(bibgolfadas)

# Passo 1: Definir os parâmetros do poço
well_params = {
    'ALFAgw': 0.0188,
    'D': 0.152,
    'Da': 0.140,
    'Dt': 0.150,
    'g': 9.81,
    'Hpdg': 1117,
    'Ht': 1279,
    'Hvgl': 916,
    'L': 4497,
    'La': 1118,
    'Lt': 1639,
    'M': 18.0,
    'R': 8314,
    'Rol': 899.996,
    'Romres': 891.9523,
    'teta': np.pi / 4,
    'T': 298,
    'mlstill': 7.1098207999143210e+02,
    'Cg': 2.3460789880222731e-05,
    'Cout': 5.8137935670717683e-03,
    'Veb': 9.0159519351419036e+01,
    'Epsilon': 3.5822558226225404e-02,
    'Kw': 1.0212053760436238e-03,
    'Ka': 1.7666249329670688e-04,
    'Kr': 2.4671647300003357e+02,
    'Vr': 4497*np.pi*0.1524*0.1524/4,
    'ssp0': [8912.57167477, 2222.72000107, 20819.60180571, 2193.67929049, 1425.20603112,13349.50962993],
    'Ck0': 5,
    'GL0': 1.65e5,  # Gaslift inflow [m³/d]
    'Ps0' : 1013250,  # Separator pressure [Pa]
    'Pr0' : 22500000, # Reservoir Pressure [Pa]
}

# Passo 2: Inicializar o poço
poço = bibgolfadas.Well()

# Passo 3: Adicionar um modelo FOWM ao poço
poço.adicionar_modelo(nome_modelo='FOWM', modelo_tipo='FOWM', params=well_params)

# Passo 4: Verificar se o modelo foi corretamente adicionado e inicializado
fowm_params = poço.get_modelo_params('FOWM')
print("Parâmetros do modelo FOWM:", fowm_params)

# Parâmetros de simulação
GL = 1.65e5  # Gaslift inflow [m³/d]
fat_Wgc = 101325 * well_params['M'] / (293 * well_params['R']) / 3600 / 24
Wgc = GL * fat_Wgc # Gaslift inflow [kg/s]
Ps = 1013250  # Separator pressure [Pa]
Pr = 2.25e7   # Reservoir Pressure [Pa]

# Gerar a função de entrada dinâmica da choke
z_samp = [5,5]#,8,10,10,11,12,12,12,13,15,14,14,13,17,15,18,18,20,20,19,19,19,21,17,15,13,15,17]
time_step = 10 # horas

u_sim, tArray_sim = generate_dynamic_input([z_samp,z_samp], time_step)
z_sim = u_sim[0]
# Configurando o vetor de entrada para teste
X_test = [tArray_sim, z_sim, Wgc, Pr, Ps]

# Utiliza o modelo FOWM para gerar o "PDG" de referência
modelo_fowm = poço.get_modelo('FOWM')
x_, y_ = modelo_fowm.model.predict(X_test, ['t','z','Ppdg'])

# Utiliza o modelpqpinicio do you
X_train = pd.DataFrame(data={'z': y_['z']}, index=y_['t'])
Y_train = pd.DataFrame(data={'Ppdg': y_['Ppdg']}, index=y_['t'])

fig, axs = plt.subplots()
axs.plot(Y_train/1.e5)
axs_ = axs.twinx()
axs_.plot(X_train*100,'r')
plt.show()

modelo_fowm.build_bifurcation()

X_train_resampled = transformar_entrada(X_train, freq_atual='1s', freq_destino='1min')
X_train_resampled_lagged = transformar_entrada(X_train, freq_atual='1s', freq_destino='1min',num_lags=10)
Y_train_resampled = transformar_entrada(Y_train, freq_atual='1s', freq_destino='5min')
Y_train_resampled_lagged = transformar_entrada(Y_train, freq_atual='1s', freq_destino='5min',num_lags=10)

# Adiciona múltiplos modelos LSTM ao poço
lstm_config_1 = {
    'input_shape': X_train.shape,  #
    'num_classes': 1,
    'epochs': 50,
    'activation': 'tanh',
    'optimizer': 'adam',
    'loss': 'mean_squared_error'  # Ajustado para uma tarefa de regressão
}

X_train_normalizado, Y_train_normalizado, scaler_X, scaler_Y = normalizar_zscore(X_train, Y_train)
# Supondo que seus dados de entrada X_train_reshaped precisam ser ajustados
n_samples = X_train_normalizado.shape[0]  # Número de amostras
# Redimensionando os dados de entrada
X_train_normalizado = np.reshape(X_train_normalizado,(n_samples, 100, 1))


poço.adicionar_modelo(nome_modelo='ModeloLSTM1', modelo_tipo='LSTM', **lstm_config_1)
# Treina o modelo LSTM
poço.fit('ModeloLSTM1', X_train_normalizado, Y_train_normalizado, X_train_normalizado, Y_train_normalizado)
poço.fit('ModeloLSTM1', X_train_normalizado, Y_train_normalizado, X_train_normalizado, Y_train_normalizado)
y_predito = poço.get_modelo('ModeloLSTM1').model.predict(X_train_normalizado)  # Predições do modelo

# Plotando os dados
plt.figure(figsize=(10, 6))
# Gráfico dos dados reais de referência
plt.plot(Y_train_normalizado, label='Dados de Referência', linestyle='-', color='blue', alpha=0.6)
# Gráfico dos dados preditos pelo modelo
plt.plot(y_predito, label='Modelo Predito (LSTM)', linestyle='--', color='red', alpha=0.8)

plt.xlabel('Índice')
plt.ylabel('Valor de Saída (Ppdg)')
plt.title('Comparação entre Dados de Referência e Predição do Modelo LSTM')
plt.legend()
plt.grid(True)
plt.show()

