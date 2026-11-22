"""Fixed S10-only protocol. Catalog is an isolated read-only loopback test fixture."""
import json
import random
import threading
import time
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, HTTPServer

SEEDS = (91411, 91412, 91413)
VARIANTS = ('config', 'report_integrity', 'catalog_policy')
PATH = '/catalog?target-nf-type=UDM'


def schedule_for(seed):
    if seed not in SEEDS:
        raise ValueError("Unregistered acquisition seed")
    rng = random.Random(seed)
    variants = list(VARIANTS)
    rng.shuffle(variants)
    starts = [anchor + rng.randrange(300) for anchor in (350, 1450, 2680)]
    return [(w, w+30, 'S10', 4, 0, mode) for w,mode in zip(starts,variants)]


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != PATH:
            self.send_error(404)
            return
        data = b'{"test_catalog":true,"entries":[]}'
        self.send_response(200)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


class Catalog:
    def __init__(self):
        self.server = HTTPServer(('127.0.0.1', 0), Handler)
        self.worker = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.worker.start()

    def measure(self):
        conn = HTTPConnection('127.0.0.1', self.server.server_port, timeout=3)
        started = time.time()
        try:
            conn.request('GET', PATH)
            response = conn.getresponse()
            body = response.read()
            if response.status != 200 or not json.loads(body)['test_catalog']:
                raise ValueError('Incomplete catalog observation')
            at = time.time()
            return dict(method='GET', path=PATH, status=response.status, started=started,
                        completed=at, latency_ms=(at-started)*1000, body=json.loads(body),
                        request_bytes=len(('GET '+PATH+' HTTP/1.1\r\n').encode()),
                        response_bytes=len(body), error=None, endpoint='isolated_empty_test_catalog')
        finally:
            conn.close()

    def close(self):
        self.server.shutdown()
        self.worker.join()
        self.server.server_close()
