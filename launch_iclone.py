"""Launch AnyMove -> iClone directly from iClone 8 Script > Load Python.

Choose F:\anymove\launch_iclone.py. Nothing is written until the user
explicitly approves the in-app Import action. This is for the exact Dragon
same-rig profile, not a generic Auto-Retargeter.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0,_ROOT)

from iclone_adapter.run_in_iclone import show

show()
