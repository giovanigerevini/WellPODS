import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import re

class CLAV():
    
    def __init__(self,):   
        # Dicionário de operadores de escritas dos modelos. Pode ser modificado ao carregar a classe.
        self.operadores = ['=', '+', '-', '/', '*', '(', ')', 'sqrt', ';', '))', '((', 'sqrt(', 'sen(']
             
        self.modelo = '', 
        self.variaveisMedidas = None
        self.constantesModelo = None
        self.equacoes = None
        self.variaveis = None
        self.variaveisNaoMedidas = None
        self.matrizOcorrencia = None
        self.equacoes = None
        self.variaveisNMedidasObservaveis = None
        self.variaveisIndeterminaveis = None
        self.grauSobredeterminacao = None
        self.equacoesNUtilizadas = None
        self.variaveisdeCorte = []
        self.variaveisdeCorteDeterminaveis = None
        self.variaveldeCorteIndeterminavel = None

    def Formulacao_Modelo(self, modelo, variaveisMedidas, constantesModelo, log=None):
        """
        Processa um arquivo de modelo de equações, identificando variáveis, variáveis medidas e constantes.

        Parâmetros:
        - modelo: str
            Caminho para o arquivo em formato `.txt` contendo as equações do modelo. 
            O arquivo deve ser formatado com operadores separados por espaços, como: 'a = b * x + sqrt( d ) / e'.
        
        - variaveisMedidas: list
            Lista contendo as variáveis que são medidas diretamente no modelo. Essas variáveis serão removidas da lista de variáveis a serem determinadas.
        
        - constantesModelo: list
            Lista contendo as constantes do modelo, que também serão removidas da lista de variáveis a serem determinadas.
        
        - log: bool ou None, opcional
            Se fornecido (ou seja, diferente de `None`), imprime o processo de carregamento do modelo, exibindo o número de equações, variáveis, variáveis medidas e não medidas.
        
        Funcionalidade:
        - Carrega um arquivo de equações em formato `.txt` e processa cada equação para identificar as variáveis e operadores.
        - Remove operadores, constantes e variáveis medidas da lista de variáveis do modelo.
        - Gera uma lista de variáveis que ainda precisam ser determinadas (variáveis não medidas).
        - Se o parâmetro `log` for fornecido, exibe o progresso e detalhes do modelo.

        Comportamento:
        1. **Carregamento das equações**: O arquivo de modelo é lido utilizando `pandas`, e as equações são armazenadas em um DataFrame. 
        2. **Identificação de variáveis**: As equações são analisadas, e todas as variáveis são extraídas, ignorando os operadores definidos.
        3. **Filtragem de variáveis**: As variáveis medidas e as constantes fornecidas são removidas da lista de variáveis. O restante é tratado como variáveis não medidas.
        4. **Ordenação**: As variáveis restantes e as não medidas são ordenadas alfabeticamente para facilitar a análise.
        5. **Logging opcional**: Caso `log` seja fornecido, o processo de modelagem é registrado com o número de equações, variáveis e variáveis não medidas.

        Retorno:
        - Equacoes: pd.DataFrame
            O DataFrame contendo as equações carregadas do arquivo de modelo.
        - variaveis: list
            A lista de variáveis restantes após a remoção das constantes e variáveis medidas, ordenadas alfabeticamente.
        - variaveisNaoMedidas: list
            A lista de variáveis que ainda não foram medidas, ordenadas alfabeticamente.

        Exemplo de uso:
        >>> Equacoes, variaveis, variaveisNaoMedidas = Formulacao_Modelo("modelo.txt", ["x", "y"], ["a", "b"], log=True)

        Exceções:
        - Lança uma mensagem de erro genérica caso o arquivo não seja encontrado ou ocorra algum problema no processamento do modelo.

        """
        self.modelo = modelo
        
        try:
            # Carrega o modelo
            Equacoes = pd.read_csv(self.modelo, header=None, names=["Equações"])
            Equacoes.index = Equacoes.index + 1

            # Identifica variáveis
            variaveis = set()  # Usar set diretamente para evitar duplicatas
            for eq in Equacoes["Equações"]:
                # Dividir a equação por espaço e verificar cada token
                varAux = eq.split()
                variaveis.update([w for w in varAux if w not in self.operadores])

            # Remover operadores e strings vazias das variáveis
            variaveis = {w for w in variaveis if w and w not in self.operadores}
            
            # Lista de variáveis não medidas inicialmente contém todas as variáveis
            variaveisNaoMedidas = variaveis.copy()

            # Remover variáveis medidas e constantes do modelo
            variaveisNaoMedidas.difference_update(variaveisMedidas)
            variaveisNaoMedidas.difference_update(constantesModelo)
            
            # Remover variáveis medidas e constantes do conjunto de todas as variáveis
            variaveis.difference_update(constantesModelo)
            
            # Converter sets de volta para listas e ordenar
            variaveis = sorted(variaveis)
            variaveisNaoMedidas = sorted(variaveisNaoMedidas)

        except Exception as e:
            print(f"Erro ao processar o modelo: {str(e)}")

        # Logging opcional
        if log is not None:
            print(f"Modelo {modelo} carregado com sucesso")
            print(f"Número de equações: {len(Equacoes)}")
            print(f"Número de variáveis: {len(variaveis)}")
            print(f"Número de variáveis medidas: {len(variaveisMedidas)}")
            print(f"Número de variáveis não medidas: {len(variaveisNaoMedidas)}")
            print("==============================================================")
            print("Variáveis:")
            print(variaveis)
            print("==============================================================")
            print("Variáveis medidas:")
            print(variaveisMedidas)
            print("Variáveis não medidas:")
            print(variaveisNaoMedidas)
            print("==============================================================")
            print(Equacoes)
            print("==============================================================")

        self.equacoes = Equacoes
        self.variaveis = variaveis
        self.variaveisNaoMedidas = variaveisNaoMedidas

    def visualiza_matriz(self, matrizOcorrencia):
        # sns.set(rc={'figure.figsize':(len(variaveis)*2, len(equacoes)*1)})
        sns.set(rc={'figure.figsize':(18,12)})
        # Ordena a matriz de acordo com o índice
        matrizOcorrenciaVisualiza = matrizOcorrencia.sort_index(axis=1)
        
        # Define o índice visualiza (supondo que Equacoes seja uma variável global ou definida anteriormente)
        if self.variaveisNMedidasObservaveis is not None:  # Verifica se a lista está vazia
            indiceVisualiza = [''.join([c + '\u0336' for c in self.equacoes.iloc[i-1].values[0]]) 
                                if i in self.variaveisNMedidasObservaveis['eq'].values 
                        else self.equacoes.iloc[i-1].values[0] for i in self.equacoes.index]
        else:
            indiceVisualiza = [self.equacoes.iloc[i-1].values[0] for i in self.equacoes.index]

        matrizOcorrenciaVisualiza.index = indiceVisualiza
        
        # Define um colormap para destacar o valor 2 como vermelho
        cmap = ListedColormap(['white', 'black', 'red'])
        
        # Cria o heatmap, com anotação dos valores e cbar desativado
        sns.heatmap(matrizOcorrenciaVisualiza,
                    annot=True,
                    cbar=False,
                    linewidths=0.25,
                    cmap=cmap,
                    vmin=0, vmax=2)  # Define os limites de valores para o colormap

        plt.show()

    def Matriz_de_Ocorrencia(self, visualiza=None):
        
        matrizOcorrencia = pd.DataFrame(0, columns = self.variaveis, index = self.equacoes.index)
            
        for i in matrizOcorrencia.index-1:
            varAux = self.equacoes.iloc[i].values[0].split()
            for w in varAux:
                try:
                    matrizOcorrencia.iloc[i][w] = 1.0
                except:
                    None
        
        if visualiza:
            self.visualiza_matriz(matrizOcorrencia)
            
        return matrizOcorrencia    

    def Identifica_Variaveis_Medidas(self, matrizOcorrencia, visualiza=None):
        """
        Identifica variáveis medidas na matriz de ocorrência.
        
        matrizOcorrencia: DataFrame contendo a matriz de ocorrência.
        variaveisMedidas: Lista de variáveis que foram medidas.
        visualiza: Função opcional para visualização da matriz.
        """
        novamatrizOcorrencia = matrizOcorrencia.copy()
        # Verifica se as variáveis medidas estão presentes na matriz
        for var in self.variaveisMedidas:
            if var in novamatrizOcorrencia.columns:
                # Define a coluna correspondente à variável medida com zeros
                novamatrizOcorrencia[var] = 0.0
            else:
                print(f"Variável '{var}' não encontrada na matriz de ocorrência. Revise a lista de variáveis medidas.")

        # Se fornecido, chama a função de visualização
        if visualiza:
            self.visualiza_matriz(visualiza)

        return novamatrizOcorrencia
   
    def Determina_Ordem_Preliminar(self, matrizOcorrencia, visualiza=None, log=None):
        """
        Determina a ordem preliminar de variáveis determináveis em um sistema de equações baseado em uma matriz de ocorrência.

        Parâmetros:
        - matrizOcorrencia: pd.DataFrame
            A matriz de ocorrência, onde as linhas representam equações e as colunas representam variáveis.
            Um valor 1 na célula indica que a variável está presente na equação correspondente.
        
        - variaveisNMedidasObservaveis: pd.DataFrame
            Um DataFrame contendo as variáveis que já foram identificadas como não medidas e observáveis.
            As variáveis determináveis nesta função serão adicionadas a este DataFrame.
        
        - visualiza: bool ou None, opcional
            Se fornecido, a função irá gerar uma visualização da matriz de ocorrência modificada.
        
        - log: bool ou None, opcional
            Se fornecido, a função irá imprimir mensagens de progresso e relatórios de log durante o processo de determinação.

        Funcionalidade:
        - A função realiza iterações sobre a matriz de ocorrência, procurando por variáveis que possam ser determinadas diretamente (variáveis presentes em apenas uma equação).
        - As variáveis identificadas são registradas no DataFrame `variaveisNMedidasObservaveis` com a respectiva equação que as determinou.
        - A matriz de ocorrência é atualizada removendo variáveis à medida que elas são determinadas, e a função continua até que não haja mais variáveis unitárias.

        Retorno:
        - novaMatrizOcorrencia: pd.DataFrame
            A matriz de ocorrência modificada, com as variáveis determináveis já zeradas.
        
        - variaveisNMedidasObservaveis: pd.DataFrame
            O DataFrame atualizado contendo todas as variáveis não medidas que foram observadas, juntamente com as equações que as determinaram.

        Exemplo de uso:
        >>> novaMatriz, variaveisObservaveis = Determina_Ordem_Preliminar(matrizOcorrencia, variaveisNaoMedidas, visualiza=True, log=True)
        """
        # Cópia da matriz de ocorrência
        novaMatrizOcorrencia = matrizOcorrencia.copy()
        
        # Substituição de valores '2' por '0' na matriz auxiliar
        matrizOcorrenciaAux = matrizOcorrencia.replace(2, 0)
        
        # Loop para identificar variáveis unitárias
        while True:
            # Identifica as linhas com soma igual a 1 (variáveis determináveis)
            linhasUnitarias = matrizOcorrenciaAux[matrizOcorrenciaAux.sum(axis=1) == 1]

            # Se não houver mais variáveis determináveis, interrompe o loop
            if linhasUnitarias.empty:
                break
            
            # Identifica a primeira equação e a variável determinável
            equacao = linhasUnitarias.index[0]
            variavelDeterminavel = linhasUnitarias.loc[equacao][linhasUnitarias.loc[equacao] == 1].index[0]
            
            # Adiciona a variável determinável atualizando o DataFrame de variáveis não medidas observáveis
            novaVariavel = pd.DataFrame([[variavelDeterminavel, equacao]], columns=['var', 'eq'])
            self.variaveisNMedidasObservaveis = pd.concat([self.variaveisNMedidasObservaveis, novaVariavel], ignore_index=True)
            
            # Zera a coluna da variável determinável nas duas matrizes
            novaMatrizOcorrencia[variavelDeterminavel] = 0
            matrizOcorrenciaAux[variavelDeterminavel] = 0
            
            # Log do progresso (se ativado)
            if log:
                print(f'A variável {variavelDeterminavel} é determinada pela equação {equacao}')
        
        # Verificação da completude do sistema
        if log:
            if matrizOcorrenciaAux.sum().sum() == 0:
                print("==============================================================")
                print('Sistema totalmente determinável com a ordem pré-estabelecida')
                print("==============================================================")
            else:
                print("==============================================================")
                print('Continuar a determinação da ordem de precedência do modelo construído')
                print("==============================================================")
        
        # Visualiza a matriz resultante (se ativado)
        if visualiza:
            self.visualiza_matriz(novaMatrizOcorrencia, visualiza)
            plt.show()
        
        return novaMatrizOcorrencia

    def Classificacao_Variaveis(self, log=None):
        """
        Classifica variáveis em indetermináveis, removendo as variáveis medidas e as variáveis não medidas observáveis.
        
        Parâmetros:
        - variaveis: list
            Lista contendo todas as variáveis presentes no modelo.
        
        - variaveisMedidas: list
            Lista contendo as variáveis medidas diretamente. Essas variáveis serão removidas da lista de indetermináveis.
        
        - variaveisNMedidasObservaveis: pd.DataFrame
            DataFrame contendo variáveis não medidas observáveis, geralmente com uma coluna 'var' contendo o nome das variáveis.
        
        - log: bool ou None, opcional
            Se fornecido, imprime informações adicionais durante o processo de classificação.

        Funcionalidade:
        - A função remove as variáveis medidas e as variáveis não medidas observáveis da lista de todas as variáveis, deixando apenas as variáveis que não podem ser determinadas diretamente (variáveis indetermináveis).
        
        Retorno:
        - variaveisIndeterminaveis: list
            Lista de variáveis indetermináveis restantes após a remoção das variáveis medidas e não medidas observáveis.

        Exemplo de uso:
        >>> indeterminaveis = Classificacao_Variaveis(variaveis, variaveisMedidas, variaveisNMedidasObservaveis, log=True)
        """
        # Cria uma lista de variáveis indetermináveis, inicialmente contendo todas as variáveis
        variaveisIndeterminaveis = list(self.variaveis)
        
        # Extrai a coluna 'var' do DataFrame de variáveis não medidas observáveis
        variaveisNMedidasObservaveis = list(self.variaveisNMedidasObservaveis['var'].values)
        
        # Converte para conjuntos para melhorar a eficiência na remoção
        variaveisIndeterminaveis = set(variaveisIndeterminaveis)
        
        # Remove variáveis não medidas observáveis e variáveis medidas
        variaveisIndeterminaveis.difference_update(variaveisNMedidasObservaveis)
        variaveisIndeterminaveis.difference_update(self.variaveisMedidas)
        
        # Converte de volta para lista e ordena
        variaveisIndeterminaveis = sorted(variaveisIndeterminaveis)
        self.variaveisIndeterminaveis = variaveisIndeterminaveis

        # Log opcional para imprimir informações adicionais
        if log is not None:
            print(f'Nº de Variáveis Indetermináveis: {len(variaveisIndeterminaveis)}')
            
            if len(variaveisIndeterminaveis) > 0:
                print('Variáveis Indetermináveis: ' + str(variaveisIndeterminaveis))
            
            print("Variáveis não medidas observáveis:")
            print(variaveisNMedidasObservaveis)
      
    def Variaveis_Redundantes():
        pass
    
    def Grau_Sobredeterminacao(self, log=None):
        """
        Calcula o grau de sobredeterminação de um sistema de equações, identificando quantas equações ainda não foram utilizadas.

        Parâmetros:
        - variaveisNMedidasObservaveis: pd.DataFrame
            DataFrame contendo as variáveis não medidas observáveis, geralmente com uma coluna 'eq' indicando as equações que já foram utilizadas.
        
        - Equacoes: pd.DataFrame
            DataFrame contendo todas as equações do sistema. O índice do DataFrame representa o número da equação.
        
        - log: bool ou None, opcional
            Se fornecido, imprime o grau de sobredeterminação e as equações que não foram utilizadas.

        Funcionalidade:
        - A função calcula quantas equações ainda não foram utilizadas no processo de determinação das variáveis observáveis.
        - O grau de sobredeterminação é o número de equações que não foram utilizadas para determinar nenhuma variável.
        
        Retorno:
        - grauSobredeterminacao: int
            O número de equações que ainda não foram utilizadas (grau de sobredeterminação).
        
        - equacoesNUtilizadas: list
            Uma lista contendo os índices das equações que não foram utilizadas.

        Exemplo de uso:
        >>> grau, equacoesNaoUtilizadas = Grau_Sobredeterminacao(variaveisNMedidasObservaveis, Equacoes, log=True)
        """
        # Lista de todas as equações disponíveis no sistema
        equacoesNUtilizadas = list(self.Equacoes.index.values)
        
        # Lista de equações já utilizadas, extraídas de variaveisNMedidasObservaveis
        equacoesUtilizadas = list(self.variaveisNMedidasObservaveis['eq'].values)
        
        # Utiliza um conjunto para melhorar a eficiência na remoção de equações utilizadas
        equacoesNUtilizadas = set(equacoesNUtilizadas)
        
        # Remove as equações que já foram utilizadas
        equacoesNUtilizadas.difference_update(equacoesUtilizadas)
        
        # Converte de volta para lista e ordena
        equacoesNUtilizadas = sorted(equacoesNUtilizadas)
        
        # Calcula o grau de sobredeterminação (quantidade de equações não utilizadas)
        grauSobredeterminacao = len(equacoesNUtilizadas)

        # Log opcional para exibir informações detalhadas
        if log is not None:
            print('Grau de Sobredeterminação: ' + str(grauSobredeterminacao))
            
            if grauSobredeterminacao > 0:
                print('Equações não utilizadas: ' + str(equacoesNUtilizadas))

        self.grauSobredeterminacao = grauSobredeterminacao
        self.equacoesNUtilizadas = equacoesNUtilizadas

    def Escolha_Variavel_Corte(self, matrizOcorrencia, visualiza=None, log=None):
        """
        Escolhe a variável de corte com base na matriz de ocorrência e atualiza a matriz para refletir essa escolha.
        
        Parâmetros:
        - matrizOcorrencia: pd.DataFrame
            DataFrame que representa a matriz de ocorrência, onde as linhas são equações e as colunas são variáveis.
            A matriz deve conter valores que indicam a presença (1) ou ausência (0) de variáveis nas equações.
        
        - variaveisdeCorte: list
            Lista que contém as variáveis de corte já escolhidas. A variável selecionada será adicionada a esta lista.
        
        - visualiza: bool ou None, opcional
            Se fornecido, gera uma visualização da matriz de ocorrência atualizada após a escolha da variável de corte.
        
        - log: bool ou None, opcional
            Se fornecido, imprime o nome da variável escolhida como variável de corte.

        Funcionalidade:
        - A função identifica a variável de corte com maior ocorrência na matriz de ocorrência.
        - Atualiza a matriz para refletir que a variável foi escolhida como variável de corte, substituindo valores 1 por 2.
        - Adiciona a variável escolhida à lista de variáveis de corte.

        Retorno:
        - novaMatrizOcorrencia: pd.DataFrame
            A matriz de ocorrência atualizada, onde as variáveis de corte são marcadas com o valor 2.
        
        - variaveisdeCorte: list
            A lista de variáveis de corte atualizada, contendo a nova variável escolhida.

        Exemplo de uso:
        >>> novaMatriz, variaveisAtualizadas = Escolha_Variavel_Corte(matrizOcorrencia, variaveisdeCorte, visualiza=True, log=True)
        """
        # Criação de cópias da matriz de ocorrência para evitar modificar o original
        novaMatrizOcorrencia = matrizOcorrencia.copy()
        novaMatrizOcorrenciaAux = novaMatrizOcorrencia.replace(2, 0)  # Substitui valores '2' por '0' na matriz auxiliar
        
        # Identifica a coluna com a maior soma (variável de maior ocorrência)
        colunaMaiorOcorrencia = novaMatrizOcorrenciaAux.sum(axis=0)
        colunaMaiorOcorrencia = colunaMaiorOcorrencia[colunaMaiorOcorrencia == colunaMaiorOcorrencia.max()]
        
        # Seleciona a primeira variável com maior ocorrência como variável de corte
        variavelDeCorte = colunaMaiorOcorrencia.index[0]
        
        # Marca a variável de corte na matriz de ocorrência substituindo 1 por 2
        novaMatrizOcorrencia[variavelDeCorte] = novaMatrizOcorrencia[variavelDeCorte].replace(1, 2)
        
        # Adiciona a variável de corte à lista de variáveis de corte
        self.variaveisdeCorte.append(variavelDeCorte)
        
        # Visualiza a matriz de ocorrência atualizada, se solicitado
        if visualiza:
            self.visualiza_matriz(novaMatrizOcorrencia)
        
        # Logging opcional para exibir a variável de corte escolhida
        if log:
            print(f'A variável de corte determinada é {variavelDeCorte}')
                
        return novaMatrizOcorrencia 

    def Iteracao_de_Escolha_Variaveis_Corte(self, matrizOcorrencia, visualiza=None, log=None):
        """
        Itera sobre a matriz de ocorrência para determinar a ordem de escolha de variáveis de corte,
        até que todas as variáveis possíveis sejam determinadas.

        Parâmetros:
        - matrizOcorrencia: pd.DataFrame
            A matriz de ocorrência onde as linhas representam equações e as colunas representam variáveis.
            Cada célula contém 1 se a variável está presente na equação, 2 se a variável já foi processada, e 0 se ausente.
        
        - equacoes: pd.DataFrame
            O DataFrame contendo todas as equações do sistema, com o índice representando o número da equação.
        
        - variaveisNMedidasObservaveis: pd.DataFrame
            O DataFrame contendo variáveis não medidas observáveis, com uma coluna 'eq' que indica as equações que já foram utilizadas para determinar as variáveis.
        
        - variaveisdeCorte: list, opcional
            Lista de variáveis de corte já escolhidas. A lista é atualizada ao longo das iterações com novas variáveis de corte. (Padrão: lista vazia)
        
        - visualiza: bool ou None, opcional
            Se fornecido, gera uma visualização da matriz de ocorrência a cada iteração.
        
        - log: bool ou None, opcional
            Se fornecido, imprime o progresso e as variáveis de corte escolhidas a cada iteração.

        Funcionalidade:
        - A função realiza múltiplas iterações de escolha de variáveis de corte até que todas as variáveis determináveis tenham sido processadas.
        - Em cada iteração, a função tenta identificar quais variáveis podem ser determinadas e, em seguida, escolhe uma variável de corte para continuar o processo.
        - Ao final, a função retorna a matriz de ocorrência atualizada, as variáveis não medidas observáveis e as variáveis de corte.

        Retorno:
        - matrizOcorrencia_aux: pd.DataFrame
            A matriz de ocorrência atualizada após o processo de iteração.
        
        - variaveisNMedidasObservaveis: pd.DataFrame
            O DataFrame atualizado contendo variáveis não medidas observáveis, com suas equações correspondentes.
        
        - variaveisdeCorte: list
            A lista de variáveis de corte atualizada, contendo todas as variáveis de corte escolhidas ao longo do processo.

        Exemplo de uso:
        >>> matrizAtualizada, variaveisObservaveis, cortes = Iteracao_de_Escolha_Variaveis_Corte(matrizOcorrencia, equacoes, variaveisNaoMedidas, visualiza=True, log=True)
        """
        
        # Inicializando a matriz de ocorrência auxiliar
        matrizOcorrencia_aux = matrizOcorrencia.copy()
        
        # Inicializa o controle de iteração
        c = 1
        while c > 0:
            
            # Determina a ordem preliminar, removendo variáveis já determinadas
            matrizOcorrencia_aux = self.Determina_Ordem_Preliminar(matrizOcorrencia_aux,
                                                                   visualiza=False,
                                                                   log=log)
            
            # Verifica se ainda há variáveis a serem determinadas (soma de valores 1)
            c = matrizOcorrencia_aux.replace(2, 0).values.sum()

            if c > 0:
                # Escolhe a próxima variável de corte
                matrizOcorrencia_aux = self.Escolha_Variavel_Corte(matrizOcorrencia_aux,
                                                                   visualiza=visualiza, 
                                                                   log=log)
        
        # Atualiza Grau de Sobredeterminação
        self.Grau_Sobredeterminacao(log)

        # Visualiza a matriz resultante, se solicitado
        if visualiza:
            self.visualiza_matriz(matrizOcorrencia_aux)
        
        # Exibe mensagem de finalização
        if log:
            print('===================================')
            print('Determinação de Ordem Preliminar terminada') 
            print('===================================')    
            if self.grauSobredeterminacao > 0:
                print('Equações não utilizadas:')
                print(self.equacoesNaoUtilizadas)
                print('===================================')
            else:
                print('Todas equações utilizadas')
                print('===================================')

        return matrizOcorrencia_aux

    def elimina_Variavel_Corte(self, matrizOcorrencia, variavelDeCorte, log=None, visualiza=None):
        variaveisdeCorteDeterminaveis = []
        novaMatrizOcorrencia = matrizOcorrencia.copy()

        if self.graudeSobredeterminacao == 0:
            
            if log != None:
                
                print(f'Todas equações utilizadas. Variáveis de corte {variavelDeCorte} indetermináveis')

        else:
            matrizOcorrenciaAux = novaMatrizOcorrencia.copy()        
            matrizOcorrenciaAux = matrizOcorrenciaAux.drop(index=self.variaveisNMedidasObservaveis['eq'])
            
            # primeiro
            c = 1
            while c > 0:
                # Identifica linhas com soma igual a 2
                linhasdeCorte = matrizOcorrenciaAux.loc[matrizOcorrenciaAux.sum(axis=1) == 2]
                
                if len(linhasdeCorte) == 0:
                    c = 0  # Para o loop se não houver mais variáveis unitárias
                    
                else:
                    # Equação e variável determinável
                    equacao = linhasdeCorte.index[0]
                    variaveldeCorteDeterminavel = linhasdeCorte.iloc[0][linhasdeCorte.iloc[0] == 2].index.values[0]
                    try:
                        variaveisdeCorteDeterminaveis = np.concatenate(variaveisdeCorteDeterminaveis,variaveldeCorteDeterminavel)
                    except:
                        None
                    
                    # Adiciona a variável determinável na lista de variáveis não medidas observáveis
                    self.variaveisNMedidasObservaveis = self.variaveisNMedidasObservaveis.append(
                        pd.DataFrame(data=[[variaveldeCorteDeterminavel, equacao]], columns=['var', 'eq']),
                        ignore_index=True
                    )
                    
                    # Substitui a coluna da variável determinável por zeros
                    novaMatrizOcorrencia[variaveldeCorteDeterminavel] = 0
                    matrizOcorrenciaAux[variaveldeCorteDeterminavel] = 0
                    
                    if log:
                        print(f'A variável de corte {variaveldeCorteDeterminavel} é determinada pela equação {equacao}')
        
        self.variaveldeCorteIndeterminavel = [item for item in variavelDeCorte if item not in variaveisdeCorteDeterminaveis]
        self.variaveisdeCorteDeterminaveis = variaveisdeCorteDeterminaveis
        
        # Visualiza a matriz resultante, se solicitado
        if visualiza:
            self.visualiza_matriz(novaMatrizOcorrencia)
            
        return novaMatrizOcorrencia
        
    def Funcao_Classificacao_de_Variaveis(self, modelo, variaveisMedidas, constantesModelo, 
                                        visualiza = None, log = None, relatorio = None):
        '''
        Função de Classificação de Variáveis e determinação do Grau de Determinação do modelo
        Entradas:
            modelo           = Arquivo txt do modelo a ser classificado, .txt
            variaveisMedidas = Vetor com as variáveis medidas, str
            constatesModelo  = Vetor com as constatnes do modelo, str
            visualiza        = Se desejar visualizar as matrizes de ocorrência, 1. (Default = None)
            log              = Se desejar visualizar os logs da classificação, 1. (Default = None)
            visualiza        = Se desejar visualizar relatório final da classificação, 1. (Default = None)        
        
        Saídas:
            equacoes         = Dataframe das equações do modelo
            variaveis        = vetor com todas as variáveis do modelo
            vNM              = vetor com as Variáveis Não Medidas do modelo
            matrizOcorrencia = matriz de ocorrencia inicial do modelo
            matrizdeOcorrenciaFinal = matriz de ocorrencia final, após classificação de variáveis
            vNMO             = DataFrame com as Variáveis Não Medidas Observáveis pelo modelo e 
                            as respectivas equações determinantes de cada variável
            vC               = vetor com as variáveis de corte
            vI               = vetor com as variáveis Indetermináveis pelo modelo
            gS               = grau de sobredeterminação do modelo
            eNU              = Equações não utiliza do modelo
        '''
        print('Inicialização.....') 

        # 1 Formulação do Modelo e 2 Identificação das Variáveis
        if log:
            print('==========================================')
            print("Etapa 1 e 2: Formulação do modelo e identificação preliminar das variáveis")
            print('==========================================')
            
        equacoes, variaveis, vNM  = Formulacao_Modelo(modelo, variaveisMedidas,constantesModelo, log)
        
        # 3 Construção da Matriz de Ocorrência    
        if log:
            print('==========================================')
            print("Etapa 3 : Construção da Matriz de Ocorrência")
            print('==========================================')
        if visualiza:
            print("Matriz de Ocorrencia")
        matrizOcorrenciaOriginal = Matriz_de_Ocorrencia(equacoes, variaveis, None, visualiza)
        
        # 4 Validação da Matriz de Ocorrência
        if log:
            print('==========================================')
            print("Etapa 4 : Validação da Matriz de Ocorrência")
            print('==========================================')

        if visualiza:
            print("Matriz de Ocorrencia Validada")
        matrizOcorrenciaValidada = Identifica_Variaveis_Medidas(matrizOcorrenciaOriginal, variaveisMedidas, equacoes, None, visualiza)
        
        print('Iniciando Classificação de Variáveis.....') 

        # 5 Determinação de Ordem Preliminar de Procedência
        if log:
            print('==========================================')
            print("Etapa 5 : Determinação de Ordem Preliminar de Procedência")
            print('==========================================')

        vNMO = pd.DataFrame(data=[],columns=['var','eq'])
        MatrizOcorrenciaPreliminar, vNMO = Determina_Ordem_Preliminar(matrizOcorrenciaValidada, None, equacoes, visualiza, log)    
        
        # 6 Classificando as Variáveis
        if log:
            print('==========================================')
            print("Etapa 6 : Determinação de Ordem Preliminar de Procedência")
            print('==========================================')
        
        vI = Classificacao_Variaveis(variaveis, variaveisMedidas, vNMO, log)
        
        # 7 Determinação do Grau de Sobredeterminação  
        if log:
            print('==========================================')
            print("Etapa 7 : Determinação Preliminar de Sobredeterminação")
            print('==========================================')
    
        gS, eNU = Grau_Sobredeterminacao(vNMO, equacoes, log)
        
        # 8 Escolha da Variável de Corte (iteração)
        if log:
            print('==========================================')
            print("Etapa 8 : Escolha da Variável de Corte (iteração)")
            print('==========================================')
    
        MatrizOcorrenciaVariavesiCorte, vNMO, vC, gS, eNU = Iteracao_de_Escolha_Variaveis_Corte(MatrizOcorrenciaPreliminar, 
                                                                                                equacoes, vNMO, [], 
                                                                                                visualiza, log)    
        
        # 9 Eliminando variável de iteração
        if log:
            print('==========================================')
            print("Etapa 9 : Eliminando Variável de Corte (iteração)")
            print('==========================================')
        
        matrizdeOcorrenciaFinal, vNMO, vCD, vCI = Elimina_Variavel_Corte(MatrizOcorrenciaVariavesiCorte, vNMO, vC, gS, equacoes, log)    
        
        if visualiza:
            visualiza_matriz(matrizdeOcorrenciaFinal, equacoes, vNMO, 1)
        
        # 10 Classificação Final
        if log:
            print('==========================================')
            print("Etapa 10 : Classificação Final")
            print('==========================================')

        vI = Classificacao_Variaveis(variaveis, variaveisMedidas, vNMO, log)
        
        gS, eNU = Grau_Sobredeterminacao(vNMO, equacoes, log)
        
        print('Classificação Terminada.....') 

        if relatorio != None:
            print(' ')
            print(' ')
            print(' ')
            print('======================================================================')
            print('              Classificação de Variáveis - Relatório ')
            print('======================================================================')
            print('Entradas: ')
            print('===================================')
            print('Modelo: ' + modelo)
            print(equacoes)
            try:
                print('Modelo: ' + constantesModelo)
            except:
                None
            visualiza_matriz(matrizOcorrenciaOriginal, equacoes, vNMO, 1) 
            print('===================================')
            print('Resultados: ')
            print('===================================')
            print("Número de Variáveis: " + str(len(variaveis)))
            print("Número de Variáveis Medidas: " + str(len(variaveisMedidas)))
            print("Número de Variáveis Não Medidas: " + str(len(vNM)))
            print("Variaveis Não Medidas:")
            print(vNM)  
            visualiza_matriz(matrizOcorrenciaValidada, equacoes, vNMO, 1) 
            print('===================================')
            print("Número de Variáveis Não Medidas Observáveis: " + str(len(vNMO)))
            print("Variaveis Variáveis Não Medidas Observáveis:")
            print(vNMO)  
            print('===================================')
            print('Grau de Sobredeterminação: ' + str(gS))
            print('Equações não utilizadas: ')
            print(eNU)
            print("Número de Variáveis Indetermináveis: " + str(len(vI)))
            print("Variaveis Indetermináveis:")
            print(vI)
            print('===================================')
            print('Matriz de Ocorrência Final')
            visualiza_matriz(matrizdeOcorrenciaFinal, equacoes, vNMO, 1) 
            print(' ')
            print(' ')
            print(' ')
            
        return equacoes, variaveis, vNM, matrizdeOcorrenciaFinal, matrizdeOcorrenciaFinal, vNMO, vC, vI, gS, eNU   
        