"""Authenticated, destination-bound messages and bounded HTTP requests."""
import hashlib
import hmac
import json
import time
import urllib.request


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sign(sender, recipient, epoch, sequence, body, key):
    packet = dict(sender=sender, recipient=recipient, epoch=epoch, sequence=sequence,
                  sent_at=time.time(), body=body)
    packet['mac'] = hmac.new(bytes.fromhex(key), canonical(packet), hashlib.sha256).hexdigest()
    return packet


def verify(packet, config):
    unsigned = {k: v for k, v in packet.items() if k != 'mac'}
    key = config['incoming_keys'].get(packet['sender'])
    if key is None or packet['recipient'] != config['id'] or packet['epoch'] != config['epoch']:
        raise ValueError('Untrusted sender, recipient, or deployment epoch')
    expected = hmac.new(bytes.fromhex(key), canonical(unsigned), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, packet['mac']):
        raise ValueError('Invalid message authentication')
    if abs(time.time() - packet['sent_at']) > 30:
        raise ValueError('Expired transport message')
    if type(packet['sequence']) is not int or packet['sequence'] < 1:
        raise ValueError('Invalid sequence')


def post(url, packet, timeout=3):
    request = urllib.request.Request(url + '/message', data=canonical(packet),
                                     headers={'Content-Type': 'application/json'})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        return json.load(response)
