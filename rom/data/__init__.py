"""
rom.data - On-disk trajectory files (one HDF5 file per VarFlExI run).

``TrajectoryWriter`` streams reconstructed snapshots to disk; ``TrajectoryFile``
gives lazy read access used by the reduced-dataset builder and the evaluation.
"""

from .trajectory_file import TrajectoryFile, TrajectoryWriter, list_trajectory_files

__all__ = ["TrajectoryFile", "TrajectoryWriter", "list_trajectory_files"]
