"""
rom – Reduced-Order Modeling pipeline for VarFlExI simulation datasets.

Subpackages:
  - eulerian: Fixed Eulerian grid definition, fluid masking, and discrete curl.
  - data: HDF5 dataset ingestion, trajectory abstraction, and validation.
  - basis: State encoding, Proper Orthogonal Decomposition (POD), and rank selection.
  - opinf: Operator Inference model construction, derivative estimation, training,
           and persistence.
"""

from . import basis, data, eulerian, opinf

__all__ = [
    "basis",
    "data",
    "eulerian",
    "opinf",
]
