"""Repeat the full audit from ZIP; original evidence stays immutable."""
import sys
from experiments.robustness_5g.audit_archive import open_archive
from experiments.freshness_5g.audit import run

if __name__=='__main__':
    root=open_archive(sys.argv[1])
    try:
        run(root)
    finally:
        root.root.close()
