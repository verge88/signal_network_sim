"""Copy a previously published frozen bundle; never fit on the new run."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

def prepare(source,target,seed):
    pins=json.loads(Path(__file__).with_name('source_pins.json').read_text())[str(seed)]
    if (target/'frozen').exists(): raise FileExistsError(target/'frozen')
    for name,digest in pins['frozen_files'].items():
        if hashlib.sha256((source/'frozen'/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Frozen bundle differs from previous confirmation: '+name)
    if (source/'source_commit.txt').read_text().strip()!=pins['source_commit']:
        raise ValueError('Unexpected source experiment')
    for phase in ['train','cal','frozen']:
        shutil.copytree(source/phase,target/phase)
    (target/'frozen_origin.json').write_text(json.dumps(pins,indent=2))

if __name__=='__main__':
    prepare(Path(sys.argv[1]),Path(sys.argv[2]),int(sys.argv[3]))
