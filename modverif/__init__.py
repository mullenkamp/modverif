from modverif.evaluator import Evaluator
from modverif.station import StationEvaluator

# These are the package's public surface, not incidental imports -- `__all__` says so, and stops
# a linter reporting them as unused. Downstream code does `from modverif import StationEvaluator`.
__all__ = ['Evaluator', 'StationEvaluator']

__version__ = '0.3.0'
