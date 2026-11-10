"""Standard-library verification primitives matching the validated control protocol."""
import hashlib
import hmac
import json
import secrets
from statistics import median

MAX_AGE=.25

def sign_report(report,key):
    raw=json.dumps(report,sort_keys=True,separators=(',', ':')).encode()
    return hmac.new(key,raw,hashlib.sha256).hexdigest()

def advance(previous,residual,allowed):
    return 0.0 if allowed else max(0.0,.99*previous+min(4.0,max(-1.0,residual))-.5)

def peer_residuals(rows):
    result={}
    for row in rows:
        peers=[other['request_count'] for other in rows if other['nf_id']!=row['nf_id']]
        center=median(peers)
        scale=max(1.0,median(abs(value-center) for value in peers))
        result[row['nf_id']]=(row['request_count']-center)/scale
    return result

def authenticated(envelope,key):
    return bool(envelope and secrets.compare_digest(envelope['signature'],sign_report(envelope['report'],key)))

def fresh(c,now,key):
    e=c['delivered']
    return bool(authenticated(e,key) and e['report']['nonce']==c['nonce']
        and e['report']['instance']==c['instance'] and 0<=now-e['report']['timestamp']<=.25)

def permit_valid(envelope,key,run_id,nf_id,now,count):
    if not authenticated(envelope,key):
        return False
    r=envelope['report']
    return (r['run_id']==run_id and r['nf_id']==nf_id and r['from']<=now<=r['until']
            and count<=r['max_requests'])

def combine(base, envelope, nonce, expected, instance, key, now):
    """No labels or intervention identifiers. Unknown is distinct from alarm."""
    if envelope is None:
        return {'H': 'alarm' if base else 'normal',
                'O': 'alarm' if base else 'unknown', 'K': 'alarm' if base else 'unknown'}
    report = envelope['report']
    fresh = (report['nonce'] == nonce and report['instance'] == instance
             and 0 <= now-report['timestamp'] <= MAX_AGE)
    authenticated = secrets.compare_digest(envelope['signature'], sign_report(report, key))
    mismatch = report['config_hash'] != expected
    return {'H': 'alarm' if base else 'normal',
            'O': 'alarm' if base or (fresh and mismatch) else ('normal' if fresh else 'unknown'),
            'K': 'alarm' if base or not authenticated or (fresh and mismatch)
                 else ('normal' if fresh else 'unknown')}

class ProtocolEngine:
    def __init__(self, model, control_key, operator_key):
        self.model=model
        self.control_key=control_key
        self.operator_key=operator_key
        self.issued=set()
        self.seen=set()
        self.high={}
        self.last_fresh=None

    def step(self,row,k_state,c,grant,now):
        replay=False
        available=True
        gap=0.
        if c is not None:
            self.issued.add(c['nonce'])
            e=c['delivered']
            available=fresh(c,now,self.control_key)
            hello=c['session_hello']
            if not authenticated(hello,self.control_key):
                raise RuntimeError('Untrusted restart identity')
            available=available and e['report']['session']==hello['report']['session']
            if authenticated(e,self.control_key) and e['report']['instance']==c['instance']:
                r=e['report']
                token=(r['session'],r['sequence'])
                if token in self.seen:
                    replay=True
                    available=False
                elif r['nonce']==c['nonce'] and r['session']==hello['report']['session']:
                    replay=r['sequence']<=self.high.get(r['session'],0)
                    self.high[r['session']]=max(r['sequence'],self.high.get(r['session'],0))
                    self.seen.add(token)
                elif r['nonce'] in self.issued:
                    self.seen.add(token)
                else:
                    replay=True  # Authenticated unsolicited response, protocol incident.
                    available=False
                # A late, previously unanswered challenge remains unknown, not replay.
            if self.last_fresh is None:
                self.last_fresh=now
            if available and not replay:
                self.last_fresh=now
            gap=now-self.last_fresh
        availability=c is not None and gap>self.model['availability_seconds']
        return dict(signals=dict(replay=bool(replay),availability=bool(availability)))
