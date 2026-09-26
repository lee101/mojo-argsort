"""mojo-argsort: index sorting (argsort, lexsort, counting sort) in Mojo.

The package is importable next to the real `argsort` distribution and never
imports it, so both can be present at once.
"""

from .core import (
    SUPPORTED_DTYPES,
    argsort,
    counting_argsort,
    count_descents,
    is_sorted,
    lexsort,
)

__all__ = [
    "argsort",
    "lexsort",
    "counting_argsort",
    "count_descents",
    "is_sorted",
    "SUPPORTED_DTYPES",
]
__version__ = "0.1.0"
