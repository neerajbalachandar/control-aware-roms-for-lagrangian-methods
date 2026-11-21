"""
rom.eulerian.xdmf - ParaView (XDMF3) descriptors for trajectory files.

``write_trajectory_xdmf(h5)`` writes ``<h5>.xmf`` next to a trajectory file
(reconstructed truth *or* ROM prediction - same schema) so ParaView can open the
time series directly (3-D iso-surfaces of |omega|, Q-criterion, slices ...)
without copying data. Uses HyperSlab references into the 5-D snapshot arrays.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union

import h5py
import numpy as np


def write_trajectory_xdmf(h5_path: Union[str, Path], fields: Sequence[str] = None,
                          out_path: Union[str, Path] = None) -> Path:
    h5_path = Path(h5_path)
    out_path = Path(out_path) if out_path else h5_path.with_suffix(".xmf")
    with h5py.File(h5_path, "r") as h:
        t = h["metadata/time"][()]
        cx, cy, cz = (h[f"metadata/coords_{a}"][()] for a in "xyz")
        order = str(h["metadata"].attrs.get("axis_order", "xyz"))
        avail = [f for f in h["snapshots"].keys()]
        shape = h[f"snapshots/{avail[0]}"].shape
        completed = h["metadata/completed"][()] if "metadata/completed" in h else np.ones(len(t), bool)
    if order != "xyz":
        raise ValueError("XDMF export supports axis_order='xyz' only")
    fields = [f for f in (fields or avail) if f in avail]
    K, Nx, Ny, Nz, _ = shape
    fn = h5_path.name
    # XDMF RectMesh: dimensions are listed slowest-to-fastest (z,y,x) for VXVYVZ;
    # our arrays are (x, y, z) C-ordered (z fastest), so give coordinates as Z,Y,X swapped.
    lines = ['<?xml version="1.0" ?>', '<Xdmf Version="3.0">', "<Domain>",
             '<Grid Name="series" GridType="Collection" CollectionType="Temporal">']
    for k in range(K):
        if not completed[k]:
            continue
        lines += [f'<Grid Name="t{k}" GridType="Uniform">', f'<Time Value="{t[k]:.10g}"/>',
                  f'<Topology TopologyType="3DRectMesh" Dimensions="{Nx} {Ny} {Nz}"/>',
                  '<Geometry GeometryType="VXVYVZ">']
        # with (x,y,z) C-order, x is the slowest index -> XDMF's "X" role must get our z coords
        for name, n in (("coords_z", Nz), ("coords_y", Ny), ("coords_x", Nx)):
            lines.append(f'<DataItem Dimensions="{n}" NumberType="Float" Precision="8" Format="HDF">{fn}:/metadata/{name}</DataItem>')
        lines.append("</Geometry>")
        for f in fields:
            lines += [f'<Attribute Name="{f}" AttributeType="Vector" Center="Node">',
                      f'<DataItem ItemType="HyperSlab" Dimensions="{Nx} {Ny} {Nz} 3" Type="HyperSlab">',
                      '<DataItem Dimensions="3 5" Format="XML">'
                      f"{k} 0 0 0 0  1 1 1 1 1  1 {Nx} {Ny} {Nz} 3</DataItem>",
                      f'<DataItem Dimensions="{K} {Nx} {Ny} {Nz} 3" NumberType="Float" Precision="4" Format="HDF">{fn}:/snapshots/{f}</DataItem>',
                      "</DataItem>", "</Attribute>"]
        lines.append("</Grid>")
    lines += ["</Grid>", "</Domain>", "</Xdmf>"]
    out_path.write_text("\n".join(lines))
    return out_path
