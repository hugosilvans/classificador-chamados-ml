"""Inferência com revisão humana explícita e rejeição de vocabulário desconhecido."""
from pathlib import Path
import joblib

DEFAULT_MODEL = Path(__file__).resolve().parents[1] / 'artifacts/model.joblib'


class Predictor:
    def __init__(self, path=DEFAULT_MODEL):
        # joblib/pickle executa código ao carregar: utilizar somente artefato local confiável.
        artifact = joblib.load(path)
        self.model = artifact['model']
        self.name = artifact['name']

    def predict(self, text):
        if not isinstance(text, str) or not 5 <= len(text.strip()) <= 2000:
            raise ValueError('texto deve ser uma string com 5 a 2000 caracteres após remover espaços nas extremidades.')
        text = text.strip()
        features = self.model.named_steps['tfidf'].transform([text])
        if features.nnz == 0:
            return {'categoria': None, 'motivo': 'vocabulario_desconhecido',
                    'revisao_humana': True, 'scores': {}, 'modelo': self.name}
        probabilities = self.model.predict_proba([text])[0]
        ranking = sorted(zip(self.model.classes_, probabilities), key=lambda pair: pair[1], reverse=True)
        return {'categoria': str(ranking[0][0]), 'revisao_humana': True,
                'motivo': 'sugestao_experimental', 'modelo': self.name,
                'scores': {str(label): round(float(value), 4) for label, value in ranking},
                'aviso': 'Scores não calibrados. Não são garantia de acerto; revisão humana obrigatória.'}
