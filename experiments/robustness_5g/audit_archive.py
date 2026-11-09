"""Run the unchanged independent replay directly against an immutable ZIP.

Only derived summaries are written to a sidecar directory; all source reads
come from the original ZIP, including every file checked by SHA-256.
"""
from pathlib import Path
import sys
import zipfile

class ArchivePath(zipfile.Path):
    @property
    def mode(self): return 'r'
    def open(self,mode='r',*args,**kwargs):
        side=self.root.audit_output/self.at
        if self.at in self.root.audit_derived and side.is_file():
            if args:
                kwargs['encoding']=args[0]
                args=args[1:]
            return side.open(mode,*args,**kwargs)
        return super().open(mode,*args,**kwargs)
    def write_text(self,data,encoding='utf-8',**kwargs):
        assert self.at.endswith('_summary.json') or self.at.endswith('replay_mismatches.json'), self.at
        side=self.root.audit_output/self.at
        side.parent.mkdir(parents=True,exist_ok=True)
        result=side.write_text(data,encoding=encoding,**kwargs)
        self.root.audit_derived.add(self.at)
        return result
    def _stream(self):
        if not hasattr(self,'_text_stream') or self._text_stream.closed:
            self._text_stream=self.open('r',encoding='utf-8')
        return self._text_stream
    def read(self,size=-1): return self._stream().read(size)
    def seek(self,*args): return self._stream().seek(*args)
    def tell(self): return self._stream().tell()
    def readable(self): return True
    def writable(self): return False
    def seekable(self): return True
    @property
    def closed(self): return False
    def __iter__(self): return iter(self._stream())

def open_archive(path):
    path=Path(path)
    archive=zipfile.ZipFile(path)
    archive.audit_output=path.with_suffix('')
    archive.audit_output.mkdir(exist_ok=True)
    archive.audit_derived=set()
    return ArchivePath(archive)

if __name__=='__main__':
    from experiments.robustness_5g.audit import run
    root=open_archive(sys.argv[1])
    try: run(root)
    finally: root.root.close()
