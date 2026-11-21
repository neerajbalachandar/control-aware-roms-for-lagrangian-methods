"""
rom.reduced - The reduced dataset shared by all ROM methods.

* ``streaming_pod``  weighted method-of-snapshots POD that streams snapshots from disk
* ``dataset``        build / load ``reduced_dataset.h5`` (bases + coordinates + signals)
* ``state``          block-scaled state assembly, output maps, model interface
* ``evaluation``     method-agnostic rollout metrics and the standard figure set
"""

from .dataset import BlockSpec, ReducedDataset, ReducedTrajectory, build_reduced_dataset
from .state import OutputMap, StateAssembler, block_field_error
from .streaming_pod import PODResult, SnapshotSet, streaming_pod

__all__ = ["BlockSpec", "ReducedDataset", "ReducedTrajectory", "build_reduced_dataset", "OutputMap",
           "StateAssembler", "block_field_error", "PODResult", "SnapshotSet", "streaming_pod"]
