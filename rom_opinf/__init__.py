"""
rom_opinf - Input-driven Operator Inference ROMs for VarFlExI (FLOWUnsteady rVPM + FEniCS).

    python -m rom_opinf.train    --dataset reduced_dataset.h5 --out artifacts/opinf
    python -m rom_opinf.evaluate --model artifacts/opinf

See ``rom_opinf/README.md``.
"""

from .model import OpInfROM
from .training import OpInfConfig, load_opinf, rank_sweep, save_opinf, train_opinf

__all__ = ["OpInfROM", "OpInfConfig", "train_opinf", "rank_sweep", "save_opinf", "load_opinf"]
