"""Minimal weak-form protocols.

This package is intentionally not a symbolic algebra system. Concrete physics modules expose a
small set of canonical forms that backends lower explicitly.
"""

from femx.forms.protocol import FormKind, WeakForm

__all__ = ["FormKind", "WeakForm"]
