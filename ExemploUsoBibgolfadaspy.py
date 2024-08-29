# Inicializa o objeto Poço
poço = Well()

# Adiciona múltiplos modelos FOWM ao poço
fowm_config_1 = {'params': {'R': 8314, 'g': 9.81}}  # Parâmetros para o primeiro modelo FOWM
fowm_config_2 = {'params': {'R': 8314, 'g': 9.81}}  # Parâmetros para o segundo modelo FOWM
poço.adicionar_modelo(nome_modelo='ModeloFOWM1', modelo_tipo='FOWM', **fowm_config_1)
poço.adicionar_modelo(nome_modelo='ModeloFOWM2', modelo_tipo='FOWM', **fowm_config_2)

# Adiciona múltiplos modelos ANN ao poço
ann_config_1 = {
    'input_shape': (100,),
    'hidden_layers': [128, 64],
    'num_classes': 3,
    'activation': 'relu',
    'optimizer': 'adam',
    'loss': 'sparse_categorical_crossentropy',
    'epochs': 30,
    'batch_size': 32
}
ann_config_2 = {
    'input_shape': (100,),
    'hidden_layers': [256, 128, 64],
    'num_classes': 3,
    'activation': 'sigmoid',
    'optimizer': 'adam',
    'loss': 'categorical_crossentropy',
    'epochs': 50,
    'batch_size': 16
}
poço.adicionar_modelo(nome_modelo='ModeloANN1', modelo_tipo='ANN', **ann_config_1)
poço.adicionar_modelo(nome_modelo='ModeloANN2', modelo_tipo='ANN', **ann_config_2)

# Adiciona múltiplos modelos LSTM ao poço
lstm_config_1 = {
    'input_shape': (100, 10),
    'num_classes': 3,
    'epochs': 50,
    'activation': 'tanh',
    'optimizer': 'adam',
    'loss': 'sparse_categorical_crossentropy'
}
lstm_config_2 = {
    'input_shape': (100, 10),
    'num_classes': 3,
    'epochs': 100,
    'activation': 'relu',
    'optimizer': 'rmsprop',
    'loss': 'categorical_crossentropy'
}
poço.adicionar_modelo(nome_modelo='ModeloLSTM1', modelo_tipo='LSTM', **lstm_config_1)
poço.adicionar_modelo(nome_modelo='ModeloLSTM2', modelo_tipo='LSTM', **lstm_config_2)

# Treina todos os modelos FOWM
poço.fit('ModeloFOWM1', ssp0, tArray_sim, z_sim, Wgc_sim, Pr_sim, Ps_sim, data_observed, method='PSO')
poço.fit('ModeloFOWM2', ssp0, tArray_sim, z_sim, Wgc_sim, Pr_sim, Ps_sim, data_observed, method='TreePartizen')

# Treina todos os modelos ANN
poço.fit('ModeloANN1', X_train, y_train, X_val, y_val)
poço.fit('ModeloANN2', X_train, y_train, X_val, y_val)

# Treina todos os modelos LSTM
poço.fit('ModeloLSTM1', X_train, y_train, X_val, y_val)
poço.fit('ModeloLSTM2', X_train, y_train, X_val, y_val)

# Avalia todos os modelos e imprime os resultados
resultados = poço.avaliar_todos_modelos(X_test, y_test)
for nome_modelo, erro in resultados.items():
    print(f"Modelo: {nome_modelo}, Erro: {erro}")
