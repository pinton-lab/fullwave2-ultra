"""pytest config for the fullwave2-ultra tests/ dir.

Underscore-prefixed files (``_*.py``) are dev-only scratch harnesses (GPU +
out-of-repo data, figure generators), not collected gates. This keeps pytest
from importing/collecting them on a clean checkout.
"""
collect_ignore_glob = ["_*.py"]
