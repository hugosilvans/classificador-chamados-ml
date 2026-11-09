# Model card — classificador de chamados

## Identificação

- Tarefa: classificação multiclasse de texto em português.
- Modelo: TF-IDF de unigramas/bigramas + regressão logística (selecionada por CV).
- Estado: protótipo educacional, execução local, sem validação de produção.
- Responsável pelo portfólio: Hugo da Silva Nunes; implementação inicial assistida por IA.
- Uso pretendido: estudar um fluxo reproduzível de treinamento, avaliação e inferência.
- Uso inadequado: decisões automáticas em atendimento real, prioridade de incidentes ou avaliação de funcionários.

## Dados

110 frases sintéticas elaboradas com assistência de IA, disponíveis integralmente em `data/seed.json`. Cinco categorias com 16 exemplos de treino e 6 de teste cada. Nenhuma frase foi extraída de usuários, de uma empresa ou da universidade. Não houve coleta de chamados reais. Rótulos são didáticos, não revisados por especialistas externos.

Os conjuntos foram definidos antes da execução. A checagem de duplicatas usa trim e casefold. Isso não detecta paráfrases. A mesma fonte de escrita e a semelhança de cenários limitam a independência do teste. Para uma avaliação externa, obter novos exemplos de outra origem, devidamente autorizados, e revisar a política de rotulagem.

## Treinamento e avaliação

Semente 42; quatro folds estratificados apenas no treino. Comparação de DummyClassifier, MultinomialNB e LogisticRegression. Escolha entre os dois modelos não triviais por F1 macro médio; empate favorece NB. Nenhuma busca de hiperparâmetros ou ajuste após leitura do teste.

Os parâmetros completos estão em `src/train.py`; versões, SHA-256 dos dados, métricas por classe e matriz de confusão estão em `reports/metrics.json`. A avaliação inicial obteve 26 acertos em 30 frases (F1 macro 0,865). Resultado frágil diante do tamanho e origem da base; não é benchmark externo.

## Riscos conhecidos

- Confusão entre instalação de impressora (impressão) e instalação de programas (software).
- Falha de acesso a documentos pode ser permissão, software ou rede.
- Vocabulário novo, gírias, erros ortográficos e relatos longos podem reduzir a qualidade.
- Texto com vários problemas é forçado a uma só classe.
- Sem conhecimento causal: o modelo usa associações entre termos e rótulos.
- Scores não calibrados. Revisão humana sempre necessária.
- A rejeição por vetor vazio não detecta todos os casos fora do domínio.

## Operação

Treinar localmente; o binário não é versionado. API sem persistência de chamados e sem logs de conteúdo, vinculada a 127.0.0.1. Não expor à rede pública. Não fornecer dados pessoais ou conteúdo confidencial na demonstração. O código não contém mecanismo de anonimização automática.

## Próxima versão

Novo conjunto independente, melhor política de rótulos, avaliação de exemplos ambíguos, teste de deriva de vocabulário, calibração e um mecanismo de abstenção validado. Publicação no Hugging Face depende de preparar e revisar um artefato apropriado; não foi realizada nesta versão.
