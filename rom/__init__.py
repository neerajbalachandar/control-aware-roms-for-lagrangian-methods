"""
rom - Shared core of the VarFlExI reduced-order-modelling pipeline.

Subpackages
-----------
varflexi   read raw VarFlExI run folders (particles, loads, structure, gusts)
eulerian   Lagrangian vortex particles -> fixed Eulerian grid (torch / numpy / VIC / Julia FMM)
data       trajectory files (streaming writer, lazy reader)
reduced    streaming weighted POD, reduced dataset, state assembly, evaluation
viz        consistent figures

Method folders (separate packages): ``rom_opinf`` (Operator Inference) and
``rom_koopman`` (Koopman lifting / EDMDc).
"""

from . import data, eulerian, reduced, varflexi, viz

__all__ = ["data", "eulerian", "reduced", "varflexi", "viz"]
