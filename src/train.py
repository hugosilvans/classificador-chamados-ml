"""Seleção por CV no treino; teste separado utilizado só após a seleção."""
import csv
import hashlib
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]


def load_data(path=ROOT / 'data/seed.json'):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    rows = [(text, label, split) for label, splits in data.items()
            for split, texts in splits.items() for text in texts]
    normalized = [text.strip().casefold() for text, _, _ in rows]
    if len(normalized) != len(set(normalized)):
        raise ValueError('Textos duplicados na base; revisar antes de treinar.')
    if any(not t.strip() or s not in {'train', 'test'} for t, _, s in rows):
        raise ValueError('Texto vazio ou partição desconhecida.')
    return rows


def pipeline(classifier):
    return Pipeline([
        ('tfidf', TfidfVectorizer(strip_accents='unicode', ngram_range=(1, 2), sublinear_tf=True)),
        ('classifier', classifier),
    ])


def train(output=ROOT / 'artifacts', reports=ROOT / 'reports'):
    rows = load_data()
    x_train = [t for t, _, s in rows if s == 'train']
    y_train = [y for _, y, s in rows if s == 'train']
    x_test = [t for t, _, s in rows if s == 'test']
    y_test = [y for _, y, s in rows if s == 'test']
    candidates = {
        'dummy_prior': pipeline(DummyClassifier(strategy='prior')),
        'naive_bayes': pipeline(MultinomialNB(alpha=1.0)),
        'logistic_regression': pipeline(LogisticRegression(C=4.0, max_iter=1000, random_state=42)),
    }
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    scores = {}
    for name, model in candidates.items():
        values = cross_val_score(model, x_train, y_train, cv=cv, scoring='f1_macro')
        scores[name] = {'mean_f1_macro': float(values.mean()), 'std': float(values.std()),
                        'folds': values.tolist()}
    # Empates favorecem Naive Bayes, o primeiro candidato não trivial.
    winner = max(['naive_bayes', 'logistic_regression'], key=lambda name: scores[name]['mean_f1_macro'])
    model = candidates[winner].fit(x_train, y_train)
    predicted = model.predict(x_test)
    labels = model.classes_.tolist()
    baseline = candidates['dummy_prior'].fit(x_train, y_train).predict(x_test)
    report = {
        'dataset': '110 textos sintéticos escritos com assistência de IA; não são chamados reais',
        'dataset_sha256': hashlib.sha256((ROOT / 'data/seed.json').read_bytes()).hexdigest(),
        'train_size': len(x_train), 'test_size': len(x_test), 'seed': 42,
        'selection': '4-fold stratified CV on training only; macro F1',
        'cv': scores, 'selected_model': winner,
        'test_accuracy': accuracy_score(y_test, predicted),
        'test_f1_macro': f1_score(y_test, predicted, average='macro'),
        'baseline_test_f1_macro': f1_score(y_test, baseline, average='macro'),
        'classes': labels,
        'classification_report': classification_report(y_test, predicted, output_dict=True, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, predicted, labels=labels).tolist(),
        'versions': {'python': platform.python_version(), 'sklearn': sklearn.__version__,
                     'numpy': np.__version__, 'joblib': joblib.__version__},
    }
    output, reports = Path(output), Path(reports)
    output.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    joblib.dump({'model': model, 'name': winner, 'dataset_sha256': report['dataset_sha256']}, output / 'model.joblib')
    (reports / 'metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    with (reports / 'predictions.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['texto', 'categoria_real', 'categoria_prevista', 'correto'])
        writer.writerows((t, y, p, y == p) for t, y, p in zip(x_test, y_test, predicted))
    summary = ['# Avaliação reproduzível', '', '**Somente dados sintéticos. Não representa desempenho real.**', '',
               f'Treino: {len(x_train)} textos. Teste separado: {len(x_test)} textos.', '',
               '| Candidato | F1 macro médio — CV | Desvio entre folds |', '|---|---:|---:|']
    summary += [f"| {n} | {s['mean_f1_macro']:.3f} | {s['std']:.3f} |" for n, s in scores.items()]
    summary += ['', f'Modelo selecionado no treino: **{winner}**.', '',
                f"Teste: acurácia **{report['test_accuracy']:.3f}**, F1 macro **{report['test_f1_macro']:.3f}**.",
                f"Baseline no teste: F1 macro **{report['baseline_test_f1_macro']:.3f}**.", '',
                '## Matriz de confusão', '', 'Linhas = classe real; colunas = previsão.', '',
                '| Real / previsto | ' + ' | '.join(labels) + ' |', '|---|' + '---:|' * len(labels)]
    summary += ['| ' + label + ' | ' + ' | '.join(map(str, row)) + ' |' for label, row in zip(labels, report['confusion_matrix'])]
    summary += ['', '## Erros observados', '']
    errors = [(t, y, p) for t, y, p in zip(x_test, y_test, predicted) if y != p]
    summary += [f'- {t}: esperado `{y}`, previsto `{p}`.' for t, y, p in errors] or ['Nenhum erro nesse pequeno teste sintético; isso não comprova generalização.']
    summary += ['', 'Não ajustar o modelo usando estes erros e continuar chamando este conjunto de teste independente.',
                'Próximo passo: um novo conjunto externo, revisado por pessoas e separado por cenário/origem.', '',
                'Os textos foram escritos pela mesma fonte: há vocabulário semelhante entre partições e viés de estilo.',
                'Não há anonimização a validar porque nenhum chamado real foi usado. Não enviar dados internos ao protótipo.', '']
    (reports / 'evaluation.md').write_text('\n'.join(summary), encoding='utf-8')
    return report


if __name__ == '__main__':
    result = train()
    print(json.dumps({k: result[k] for k in ['selected_model', 'test_accuracy', 'test_f1_macro']}, indent=2))
