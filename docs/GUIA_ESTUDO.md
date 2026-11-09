# Entenda o projeto antes de apresentar

## Explicação curta

Este protótipo recebe a descrição de um chamado de TI e sugere uma categoria. Ele transforma palavras em números com TF-IDF e usa um classificador treinado com exemplos rotulados. A implementação inicial foi feita com assistência de IA e usa dados sintéticos. A avaliação separa treino e teste; toda sugestão precisa de revisão humana.

Use essa descrição como referência técnica. Não diga que implementou sozinho, validou com usuários ou usa o sistema no trabalho sem ter feito essas etapas.

## O que significa cada parte

**Treinamento:** o algoritmo encontra relações entre textos e categorias conhecidas. Leia `src/train.py`, começando em `load_data`, depois `pipeline` e `train`.

**TF-IDF:** uma representação numérica de termos. Palavras recorrentes em quase todos os textos tendem a ser menos informativas. Bigramas preservam pares como “acesso negado”.

**Regressão logística:** apesar do nome, aqui é um algoritmo de classificação. Aprende pesos para os termos e produz scores para as categorias. Não é um modelo de linguagem generativa.

**Naive Bayes:** alternativa probabilística simples para comparar. Uma comparação ajuda a justificar a escolha sem escolher uma tecnologia apenas porque é mais complexa.

**Baseline:** o dummy ignora o texto e representa uma referência mínima. Superá-lo nesta base é uma checagem inicial, não prova de utilidade real.

**Validação cruzada:** divide o treino em quatro grupos para comparar modelos. Cada rodada aprende com três grupos e avalia no restante. O teste final permanece fora dessa escolha.

**Vazamento de dados:** ocorre quando informações que deveriam estar indisponíveis participam do aprendizado. Ajustar o TF-IDF dentro do pipeline em cada fold evita usar o vocabulário da validação nessa etapa.

**Acurácia:** fração de previsões corretas. Na primeira execução, 26 de 30 exemplos sintéticos.

**F1 macro:** combina precisão e recall por classe e calcula a média entre classes. As cinco categorias têm o mesmo peso.

**Inferência:** usar o modelo já treinado em um novo texto. Leia `src/predict.py`.

**API:** permite enviar JSON e receber a sugestão. Leia `src/api.py` e teste a interface local.

## Exercícios para ganhar domínio

1. Execute todos os comandos do README e observe os arquivos gerados.
2. Leia as quatro classificações incorretas em `reports/evaluation.md`; explique por que podem ser ambíguas.
3. Envie um texto sobre um assunto fora de TI. Observe que conter palavras conhecidas pode produzir uma categoria errada.
4. Teste um chamado com dois problemas. Explique a limitação de uma classe por texto.
5. Crie um novo conjunto de avaliação independente sem editar o teste atual. Marque sua origem e discuta as diferenças.
6. Em um experimento separado, compare outra representação. Escolha no treino; não ajuste visando acertar o teste já conhecido.

## Perguntas de entrevista

- **Você treinou um LLM?** Não. Este é um classificador tradicional de texto, pequeno e executado em CPU.
- **Por que não usar uma API generativa?** Para demonstrar o treinamento e a avaliação de um modelo controlável, sem custo de inferência externa.
- **Qual foi seu papel?** Descreva o que você efetivamente estudou, executou e modificou. A versão inicial foi assistida por IA.
- **Pode colocar em produção?** Não nesta fase. Faltam dados representativos, validação externa, medidas operacionais e tratamento adequado de incerteza.
- **Qual a principal limitação?** A avaliação usa apenas 30 textos sintéticos, escritos pela mesma fonte do treino.

## Evidências para apresentar

Mostre o dataset, a seleção via CV, a matriz de confusão, os erros e um teste da API. É mais esclarecedor demonstrar uma falha e explicar sua causa provável do que apresentar apenas a acurácia.
