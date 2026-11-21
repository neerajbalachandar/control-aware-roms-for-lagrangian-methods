"""
rom_koopman - Koopman-operator ROMs for VarFlExI (FLOWUnsteady rVPM + FEniCS).

* ``lifting``  - Mauroy & Goncalves lifting identification (port of Koopman-identification-master)
* ``edmdc``    - Korda & Mezic linear (or bilinear) Koopman predictor with inputs
* ``dictionaries`` - monomial / RBF observables

    python -m rom_koopman.train    --dataset reduced_dataset.h5 --out artifacts/koopman
    python -m rom_koopman.evaluate --model artifacts/koopman

See ``rom_koopman/README.md``.
"""

from .dictionaries import MonomialDictionary, RBFDictionary
from .edmdc import EDMDcROM
from .lifting import LiftingKoopmanROM, lifting_ident_dual, lifting_ident_main
from .training import KoopmanConfig, load_koopman, save_koopman, train_koopman

__all__ = ["MonomialDictionary", "RBFDictionary", "EDMDcROM", "LiftingKoopmanROM", "lifting_ident_main",
           "lifting_ident_dual", "KoopmanConfig", "train_koopman", "save_koopman", "load_koopman"]
