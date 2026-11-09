# Avaliação reproduzível

**Somente dados sintéticos. Não representa desempenho real.**

Treino: 80 textos. Teste separado: 30 textos.

| Candidato | F1 macro médio — CV | Desvio entre folds |
|---|---:|---:|
| dummy_prior | 0.067 | 0.000 |
| naive_bayes | 0.733 | 0.085 |
| logistic_regression | 0.757 | 0.054 |

Modelo selecionado no treino: **logistic_regression**.

Teste: acurácia **0.867**, F1 macro **0.865**.
Baseline no teste: F1 macro **0.067**.

## Matriz de confusão

Linhas = classe real; colunas = previsão.

| Real / previsto | acesso | hardware | impressao | rede | software |
|---|---:|---:|---:|---:|---:|
| acesso | 6 | 0 | 0 | 0 | 0 |
| hardware | 1 | 4 | 1 | 0 | 0 |
| impressao | 0 | 0 | 6 | 0 | 0 |
| rede | 0 | 1 | 0 | 5 | 0 |
| software | 1 | 0 | 0 | 0 | 5 |

## Erros observados

- Meu computador não obtém IP pelo cabo: esperado `rede`, previsto `hardware`.
- A bateria descarrega imediatamente ao retirar da tomada: esperado `hardware`, previsto `impressao`.
- Meu teclado tem várias teclas quebradas: esperado `hardware`, previsto `acesso`.
- Não consigo concluir a instalação da atualização: esperado `software`, previsto `acesso`.

Não ajustar o modelo usando estes erros e continuar chamando este conjunto de teste independente.
Próximo passo: um novo conjunto externo, revisado por pessoas e separado por cenário/origem.

Os textos foram escritos pela mesma fonte: há vocabulário semelhante entre partições e viés de estilo.
Não há anonimização a validar porque nenhum chamado real foi usado. Não enviar dados internos ao protótipo.
