import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
from src.api import make_server
from src.predict import Predictor
from src.train import load_data, train, ROOT


class ProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        cls.metrics = train(root / 'artifacts', root / 'reports')
        cls.predictor = Predictor(root / 'artifacts/model.joblib')
        cls.server = make_server(port=0, model_path=root / 'artifacts/model.joblib')
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f'http://127.0.0.1:{cls.server.server_port}'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()
        cls.temp.cleanup()

    def request(self, path, data=None, content_type='application/json'):
        body = json.dumps(data).encode() if data is not None else None
        request = Request(self.url + path, data=body, headers={'Content-Type': content_type})
        try:
            response = urlopen(request, timeout=5)
        except HTTPError as error:
            response = error
        with response:
            return response.status, response.read()

    def test_dataset_partitions_balanced_and_disjoint(self):
        rows = load_data()
        self.assertEqual(len(rows), 110)
        for label in {label for _, label, _ in rows}:
            self.assertEqual(sum(y == label and s == 'train' for _, y, s in rows), 16)
            self.assertEqual(sum(y == label and s == 'test' for _, y, s in rows), 6)
        train_text = {t.casefold() for t, _, s in rows if s == 'train'}
        test_text = {t.casefold() for t, _, s in rows if s == 'test'}
        self.assertFalse(train_text & test_text)

    def test_vectorizer_did_not_learn_test_only_vocabulary(self):
        vectorizer = self.predictor.model.named_steps['tfidf']
        analyze = vectorizer.build_analyzer()
        rows = load_data()
        training = {term for text, _, split in rows if split == 'train' for term in analyze(text)}
        test = {term for text, _, split in rows if split == 'test' for term in analyze(text)}
        self.assertTrue(test - training)
        self.assertFalse(set(vectorizer.vocabulary_) & (test - training))

    def test_model_roundtrip_matches_report(self):
        rows = load_data()
        texts = [t for t, _, s in rows if s == 'test']
        labels = [y for _, y, s in rows if s == 'test']
        predicted = self.predictor.model.predict(texts)
        self.assertAlmostEqual(float(np.mean(predicted == labels)), self.metrics['test_accuracy'])
        self.assertEqual(sum(map(sum, self.metrics['confusion_matrix'])), 30)

    def test_repeat_training_reproduces_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            other = train(Path(tmp) / 'model', Path(tmp) / 'reports')
        self.assertEqual(self.metrics, other)

    def test_valid_prediction_always_requires_review(self):
        result = self.predictor.predict('A impressora não imprime')
        self.assertEqual(result['categoria'], 'impressao')
        self.assertTrue(result['revisao_humana'])
        self.assertAlmostEqual(sum(result['scores'].values()), 1, places=3)

    def test_unseen_vocabulary_abstains(self):
        result = self.predictor.predict('zxqv kjzxw qvvxz')
        self.assertIsNone(result['categoria'])
        self.assertTrue(result['revisao_humana'])
        self.assertEqual(result['scores'], {})

    def test_invalid_input(self):
        for text in [None, 123, [], {}, '', '    ', 'a', 'a' * 2001]:
            with self.subTest(text=str(text)[:20]), self.assertRaises(ValueError):
                self.predictor.predict(text)

    def test_http_health_and_page(self):
        status, body = self.request('/health')
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)['status'], 'ok')
        status, body = self.request('/')
        self.assertEqual(status, 200)
        self.assertIn(b'<form', body)

    def test_http_prediction(self):
        status, body = self.request('/predict', {'texto': 'A impressora não imprime'})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)['categoria'], 'impressao')

    def test_http_bad_inputs(self):
        for data in [[], {'texto': 12}, {'texto': 'a'}, {'texto': 'A impressora falhou', 'extra': True}]:
            self.assertEqual(self.request('/predict', data)[0], 400)
        self.assertEqual(self.request('/predict', {'texto': 'teste'}, 'text/plain')[0], 415)
        self.assertEqual(self.request('/predict', {'texto': 'a' * 17000})[0], 413)
        self.assertEqual(self.request('/ausente')[0], 404)

    def test_http_malformed_json(self):
        request = Request(self.url + '/predict', data=b'{bad', headers={'Content-Type': 'application/json'})
        with self.assertRaises(HTTPError) as error:
            urlopen(request, timeout=5)
        self.assertEqual(error.exception.code, 400)
        error.exception.close()


if __name__ == '__main__':
    unittest.main()
