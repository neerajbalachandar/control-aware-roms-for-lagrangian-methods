"""
rom.viz.style - One visual language for every figure of the pipeline.

* Categorical colours (validated for colour-vision deficiency, fixed order, never cycled):
  slot 1 blue, slot 2 orange, slot 3 aqua.  Methods are pinned to slots so a colour
  always means the same thing across figures: OpInf = blue, Koopman = orange,
  third model = aqua.  Data splits: train = blue, val = orange, test = aqua.
* Ground truth (FOM) is neutral ink, solid; ROM predictions are coloured *and* dashed
  (secondary encoding); the POD projection floor is a muted dotted line.
* Fields: diverging blue-grey-red (zero = grey) for signed quantities (velocity
  fluctuations, vorticity components, errors), sequential single-hue blue for
  magnitudes.  Never rainbow.
* Every figure is written as PNG (300 dpi, for slides) and PDF (vector, for papers).
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

METHOD_COLORS = {"truth": INK, "opinf": SERIES[0], "koopman": SERIES[1], "edmdc": SERIES[1],
                 "lifting": SERIES[2], "projection": MUTED}
SPLIT_COLORS = {"train": SERIES[0], "val": SERIES[1], "test": SERIES[2]}
METHOD_STYLES = {"truth": "-", "projection": ":", "opinf": "--", "koopman": (0, (6, 2, 1, 2)),
                 "edmdc": (0, (6, 2, 1, 2)), "lifting": (0, (1.5, 1.5))}

DIVERGING = LinearSegmentedColormap.from_list(
    "rom_diverging", ["#104281", "#2a78d6", "#86b6ef", "#f0efec", "#f2a3a2", "#e34948", "#9b2423"])
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "rom_sequential", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


_PRIORITY = ("truth", "projection", "lifting", "edmdc", "koopman", "opinf")


def _key(name: str):
    low = name.lower()
    for k in _PRIORITY:
        if k in low:
            return k
    return None


def method_color(name: str, i: int = 0) -> str:
    """Colour pinned to the method family (substring match), else categorical slot i."""
    k = _key(name)
    return METHOD_COLORS[k] if k else SERIES[i % 3]


def method_style(name: str):
    k = _key(name)
    return METHOD_STYLES.get(k, "--") if k else "--"


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "semibold", "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "figure.titlesize": 11,
        "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2,
        "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "lines.linewidth": 1.5, "lines.markersize": 5, "legend.frameon": False,
        "axes.prop_cycle": matplotlib.cycler(color=SERIES[:3]),
        "image.cmap": "rom_sequential", "savefig.dpi": 300, "savefig.bbox": "tight",
        "mathtext.fontset": "dejavusans",
    })
    for cm in (DIVERGING, SEQUENTIAL):
        try:
            matplotlib.colormaps.register(cm)
        except ValueError:
            pass


def savefig(fig, path: Union[str, Path], pdf: bool = True) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"))
    if pdf:
        fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path.with_suffix(".png")


apply_style()
