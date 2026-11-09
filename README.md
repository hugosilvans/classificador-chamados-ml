# Classificador de chamados de TI com Machine Learning

Um chamado chega em texto livre. Este protótipo sugere uma das cinco categorias — **acesso, rede, impressão, hardware ou software** — para apoiar uma triagem humana.

**Projeto educacional em desenvolvimento, construído com assistência de IA.** A base contém apenas exemplos sintéticos, não chamados de uma instituição. Não há integração com sistemas reais nem evidência de impacto operacional.

## O que funciona

- Pipeline de texto com TF-IDF e classificação supervisionada em Python/scikit-learn.
- Comparação de Naive Bayes, regressão logística e baseline sem aprendizado do texto.
- Seleção por validação cruzada no treino, seguida de avaliação no teste separado.
- Relatórios reproduzíveis: F1 macro, acurácia, matriz de confusão e previsões individuais.
- API HTTP local, interface de demonstração e testes automatizados.
- Rejeição de texto sem vocabulário conhecido e revisão humana em todas as sugestões.

## Resultado da execução inicial

| Medida | Resultado |
|---|---:|
| Textos de treinamento | 80 |
| Textos de teste | 30 |
| Acurácia no teste sintético | 86,7% (26/30) |
| F1 macro no teste sintético | 0,865 |

Modelo selecionado: **regressão logística**. Esses números descrevem somente os 30 exemplos sintéticos separados. A base é pequena, balanceada e escrita pela mesma fonte; não representa diversidade de chamados reais. **Não interpretar 86,7% como precisão esperada em produção.**

Veja [avaliação completa e erros](reports/evaluation.md), [métricas JSON](reports/metrics.json) e [previsões CSV](reports/predictions.csv).

## Executar em poucos minutos

Pré-requisito: **Python 3.12** e Git. Não requer chave de API, GPU ou serviço pago. A instalação inicial das bibliotecas exige internet; treinamento e inferência usam arquivos locais.

```bash
git clone https://github.com/hugosilvans/classificador-chamados-ml.git
cd classificador-chamados-ml
python -m venv .venv
```

Ative o ambiente no Windows/PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Ou no Linux/macOS:

```bash
source .venv/bin/activate
```

Depois:

```bash
python -m pip install -r requirements.txt
python -m src.train
python -m unittest discover -s tests -v
python -m src.api
```

Abra **http://127.0.0.1:8000**. Digite um exemplo fictício. O servidor usa somente a interface local e não armazena textos recebidos.

Se o PowerShell bloquear a ativação, use diretamente `.venv\Scripts\python.exe` no lugar de `python`, sem alterar a política de segurança.

## Testar a API

Endpoints: `GET /health`, `POST /predict` e `GET /` para a demonstração.

PowerShell:

```powershell
$body = @{ texto = 'A impressora não imprime o documento' } | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -ContentType 'application/json; charset=utf-8' -Body ([System.Text.Encoding]::UTF8.GetBytes($body))
```

Linux/macOS:

```bash
curl http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"texto":"A impressora não imprime o documento"}'
```

A resposta inclui `categoria`, `scores`, `modelo` e `revisao_humana: true`. Os scores são probabilidades internas **não calibradas**, não garantia de acerto. Texto fora de vocabulário retorna `categoria: null`; texto fora de domínio que contenha palavras conhecidas ainda pode receber sugestão incorreta.

Entrada: objeto JSON contendo apenas `texto`, entre 5 e 2000 caracteres após trim. Retornos: 400 para entrada inválida, 404 para rota inexistente, 413 para corpo fora do limite e 415 para formato incorreto.

## Como funciona

1. `data/seed.json` define treino e teste previamente separados, com textos diferentes.
2. A validação cruzada divide **somente o treino** em quatro partes estratificadas.
3. Cada fold ajusta o TF-IDF usando apenas sua parcela de treino, dentro de um `Pipeline`.
4. Naive Bayes e regressão logística são comparados por F1 macro; o dummy serve como referência.
5. O vencedor é treinado nos 80 exemplos e avaliado nos 30 exemplos de teste.
6. O artefato local é salvo em `artifacts/model.joblib`; a API o carrega sem retreinar.

F1 macro dá o mesmo peso a cada categoria. O teste nunca escolhe modelo ou hiperparâmetros. O modelo publicado como código mantém o treinamento restrito aos 80 exemplos para reproduzir a avaliação.

## Estrutura

| Caminho | Papel |
|---|---|
| `src/train.py` | Treinamento, comparação e relatórios |
| `src/predict.py` | Inferência e validação |
| `src/api.py` | Servidor HTTP local |
| `demo.html` | Tela de demonstração |
| `data/seed.json` | Dados sintéticos com partições explícitas |
| `tests/test_project.py` | Testes de dados, inferência e HTTP |
| `reports/` | Resultados da execução |
| [docs/MODEL_CARD.md](docs/MODEL_CARD.md) | Escopo e limitações do modelo |
| [docs/GUIA_ESTUDO.md](docs/GUIA_ESTUDO.md) | Explicação para estudar e apresentar |

## Limites e próximos experimentos

- A separação evita duplicatas exatas, mas não elimina semelhança de vocabulário e estilo entre treino e teste.
- Apenas um assunto por chamado; textos ambíguos ou com múltiplos problemas não são tratados adequadamente.
- Não há detecção geral de assunto desconhecido, calibração, avaliação externa ou validação humana dos rótulos.
- O servidor da biblioteca padrão foi escolhido para uma demonstração local simples; não é uma implantação de produção.
- Antes de uso real: coletar dados autorizados e minimizados, separar por origem/tempo, revisar rótulos, avaliar classes desbalanceadas, calibrar e medir a abstenção.
- Para hospedagem futura: servidor/framework adequado, HTTPS, autenticação, limites de requisição e observabilidade sem registrar conteúdo sensível.
- Não carregar arquivos joblib de fontes desconhecidas: esse formato pode executar código.

## Transparência de autoria e estágio

Protótipo preparado para o portfólio de Hugo da Silva Nunes com assistência de IA na implementação, nos textos sintéticos, nos testes e na documentação. O histórico registra a versão inicial; não representa meses de trabalho, pesquisa publicada ou experiência profissional com ML. A revisão, os experimentos e o domínio do código por Hugo são próximas etapas de aprendizado.

Ainda não há modelo publicado no Hugging Face, participação em competição Kaggle ou artigo científico associado. O guia de estudo é documentação técnica do projeto.

## Referência técnica

A construção do pipeline segue a orientação de [Common pitfalls — scikit-learn](https://scikit-learn.org/stable/common_pitfalls.html) sobre ajuste de transformações apenas no treino e prevenção de vazamento de dados.
