"""Servidor HTTP local de demonstração; não expor diretamente à internet."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path

from src.predict import Predictor, DEFAULT_MODEL

HTML = Path(__file__).resolve().parents[1] / 'demo.html'


def make_server(host='127.0.0.1', port=8000, model_path=DEFAULT_MODEL):
    predictor = Predictor(model_path)

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(10)

        def log_message(self, *_):
            pass  # Não registrar textos ou URLs recebidos.

        def send(self, status, payload, content_type='application/json; charset=utf-8'):
            body = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == '/':
                self.send(200, HTML.read_bytes(), 'text/html; charset=utf-8')
            elif self.path == '/health':
                self.send(200, {'status': 'ok', 'modelo': predictor.name, 'uso': 'demonstracao_local'})
            else:
                self.send(404, {'erro': 'Rota não encontrada.'})

        def do_POST(self):
            if self.path != '/predict':
                self.send(404, {'erro': 'Rota não encontrada.'})
                return
            if self.headers.get_content_type() != 'application/json':
                self.send(415, {'erro': 'Envie Content-Type: application/json.'})
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
            except ValueError:
                self.send(400, {'erro': 'Content-Length inválido.'})
                return
            if not 0 < length <= 16000:
                self.send(413, {'erro': 'Corpo deve ter de 1 a 16000 bytes.'})
                return
            try:
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict) or set(data) != {'texto'}:
                    raise ValueError('Envie somente o campo texto em um objeto JSON.')
                result = predictor.predict(data['texto'])
            except (ValueError, UnicodeDecodeError) as error:
                self.send(400, {'erro': str(error)})
                return
            except TimeoutError:
                self.send(408, {'erro': 'Tempo esgotado para receber o corpo.'})
                return
            self.send(200, result)

    return ThreadingHTTPServer((host, port), Handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    try:
        server = make_server(port=args.port)
    except FileNotFoundError:
        parser.exit(1, 'Modelo ausente. Execute primeiro: python -m src.train\n')
    print(f'Demonstração local em http://127.0.0.1:{args.port}; Ctrl+C para encerrar.')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
