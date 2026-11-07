"""Small HTTP/2 prior-knowledge client restricted to this isolated loopback lab."""
from __future__ import annotations

import ipaddress
import json
import socket
import time
import uuid
from urllib.parse import urlsplit


def loopback_url(url: str):
    parsed = urlsplit(url)
    if parsed.scheme != 'http' or not parsed.hostname:
        raise ValueError('Only explicit http loopback addresses are permitted')
    try:
        address = ipaddress.ip_address(parsed.hostname)
    except ValueError as exc:
        raise ValueError('Use a literal loopback IP, not a hostname') from exc
    if not address.is_loopback or parsed.username or parsed.password:
        raise ValueError('This experiment is restricted to loopback')
    return parsed


class H2Client:
    def __init__(self, base_url='http://127.0.0.10:7777', timeout=4.0):
        self.address = loopback_url(base_url)
        self.timeout = timeout

    def request(self, method: str, path: str, body=None):
        from h2.config import H2Configuration
        from h2.connection import H2Connection
        from h2.events import DataReceived, ResponseReceived, StreamEnded, StreamReset
        from h2.exceptions import H2Error

        if not path.startswith('/') or path.startswith('//'):
            raise ValueError('Request path must be an origin-relative path')
        data = b'' if body is None else json.dumps(body, separators=(',', ':')).encode()
        started = time.time()
        monotonic_started = time.monotonic()
        connection_id = uuid.uuid4().hex
        received = bytearray()
        status, error = 0, None
        try:
            with socket.create_connection((self.address.hostname, self.address.port or 80), self.timeout) as sock:
                sock.settimeout(self.timeout)
                connection = H2Connection(config=H2Configuration(client_side=True, header_encoding='utf-8'))
                connection.initiate_connection()
                headers = [(':method', method), (':scheme', 'http'),
                           (':authority', self.address.netloc), (':path', path)]
                if body is not None:
                    headers += [('content-type', 'application/json-patch+json' if method == 'PATCH' else 'application/json'),
                                ('content-length', str(len(data)))]
                connection.send_headers(1, headers, end_stream=not data)
                if data:
                    connection.send_data(1, data, end_stream=True)
                sock.sendall(connection.data_to_send())
                ended = False
                while not ended:
                    packet = sock.recv(65536)
                    if not packet:
                        raise ConnectionError('HTTP/2 connection closed before END_STREAM')
                    for event in connection.receive_data(packet):
                        if isinstance(event, ResponseReceived):
                            status = int(dict(event.headers).get(':status', 0))
                        elif isinstance(event, DataReceived):
                            received.extend(event.data)
                            connection.acknowledge_received_data(event.flow_controlled_length, event.stream_id)
                        elif isinstance(event, StreamEnded):
                            ended = True
                        elif isinstance(event, StreamReset):
                            raise ConnectionError(f'HTTP/2 reset {event.error_code}')
                    pending = connection.data_to_send()
                    # This client deliberately closes each connection after its
                    # one stream. Once END_STREAM was received, a peer close is
                    # valid and no final WINDOW_UPDATE/SETTINGS ACK is required
                    # for application completion. Writing it can race with FIN.
                    if pending and not ended:
                        sock.sendall(pending)
        except (OSError, ValueError, ConnectionError, H2Error) as exc:
            error = str(exc)
            # A received 200 header with a truncated body is not a successful
            # observation. Preserve the failure instead of trusting partial data.
            status = 0
        elapsed = time.monotonic() - monotonic_started
        try:
            payload = json.loads(received) if received else None
        except (ValueError, UnicodeDecodeError):
            payload = received.decode('utf-8', errors='replace')
        return {'started': started, 'completed': time.time(), 'method': method,
                'path': path, 'status': status, 'body': payload, 'error': error,
                'connection_id': connection_id, 'stream_id': 1,
                'latency_ms': elapsed * 1000, 'request_bytes': len(data),
                'response_bytes': len(received)}
