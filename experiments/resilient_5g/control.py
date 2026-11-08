"""Session-bound witness and replay fault, with an explicit restart handshake."""
import copy
import os
from pathlib import Path
import secrets
import time
from experiments.integrated_5g.control import Control as PreviousControl
from experiments.kie_independent.run import sha
from experiments.live_5g.collect import sign_report


def observer_process(pipe, config, key, instance, challenge):
    session = secrets.token_hex(16)
    hello = dict(session=session, instance=instance, challenge=challenge)
    pipe.send(dict(report=hello, signature=sign_report(hello,key)))
    sequence = 0
    while True:
        nonce = pipe.recv()
        if nonce is None:
            return
        sequence += 1
        report = dict(config_hash=sha(Path(config).read_bytes()), instance=instance,
                      nonce=nonce, timestamp=time.time(), pid=os.getpid(),
                      session=session, sequence=sequence)
        pipe.send(dict(report=report,signature=sign_report(report,key)))


class Control(PreviousControl):
    def start_observer(self):
        self.parent, child = self.ctx.Pipe()
        challenge = secrets.token_hex(16)
        self.observer = self.ctx.Process(target=observer_process,
             args=(child,str(self.path.resolve()),self.key,self.instance,challenge))
        self.observer.start()
        if not self.parent.poll(10):
            raise RuntimeError('Authenticated observer restart failed')
        self.hello = self.parent.recv()
        r = self.hello['report']
        if (r['challenge']!=challenge or r['instance']!=self.instance or
            not secrets.compare_digest(self.hello['signature'],sign_report(r,self.key))):
            raise RuntimeError('Invalid restart handshake')

    def prepare(self, window, scenario):
        super().prepare(window,scenario)
        # Held-out benign long outage: availability incidents must not be called attacks.
        if self.root.name=='test' and window % 300 == 80:
            self.observer.terminate()
            self.observer.join(5)
            self.down_until = window+8

    def observe(self, window, scenario):
        if scenario=='R' and getattr(self,'last_scenario',None)!='R':
            self.replayed = copy.deepcopy(getattr(self,'previous',None))
        result = super().observe(window,scenario)
        if scenario=='R' and result['original'] is not None:
            result['delivered'] = copy.deepcopy(self.replayed)
        result['session_hello'] = self.hello
        self.last_scenario = scenario
        return result
