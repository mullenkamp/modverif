from modverif.evaluator import Evaluator
from modverif.station import StationEvaluator

# Top-level convenience re-exports, not incidental imports -- `__all__` says so, and stops a linter
# reporting them as unused. Downstream code does `from modverif import StationEvaluator`.
#
# This is NOT the whole public surface: `metrics`, `plots`, `cyclone`, `composite`, `station`,
# `window`, ... are public and imported by module path (`from modverif.window import max_window`).
# A new module does not belong here unless it also earns a top-level shortcut.
__all__ = ['Evaluator', 'StationEvaluator']

__version__ = '0.4.0'
