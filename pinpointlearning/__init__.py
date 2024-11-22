"""
A package for analysing student exams results. Contains the modules:
- utils
"""


import matplotlib as mpl
import matplotlib.colors as mpc
import matplotlib.pyplot as plt
import scienceplots

plt.style.use("science")


HC_COLOURS_NAMED = {
    "hclightblue": "#00BED5",
    "hccoral": "#FF5A5A",
    "hcdarkblue": "#1E5DF8",
    "hcnavy": "#003088",
    "hcdarknavy": "#2E2D62",
    "hcgrey": "#676767",
}

for cname, cdef in HC_COLOURS_NAMED.items():
    mpc._colors_full_map[cname] = cdef  # pylint: disable=W0212


mpl.rcParams["axes.prop_cycle"] = mpl.cycler(
    color=["hclightblue", "hccoral", "hcdarkblue", "hcgrey"]
)

cmap = mpl.colors.LinearSegmentedColormap.from_list(
    "hccmap", ["hclightblue", "#ffffff", "hccoral"]
)

mpl.colormaps.register(cmap, name="hccmap")

cmap = mpl.colors.LinearSegmentedColormap.from_list(
    "hccmap_lblues", ["#ffffff", "hclightblue"]
)

mpl.colormaps.register(cmap, name="hccmap_lblues")


cmap = mpl.colors.LinearSegmentedColormap.from_list(
    "hccmap_dblues", ["#ffffff", "hcdarkblue"]
)
mpl.colormaps.register(cmap, name="hccmap_dblues")


def plots_ready():
    """Gives you an excuse to import pinpointlearning unused.
    Verifies the import has succeeded and plt configs set.

    Returns:
        bool: True to verify package has imported
    """
    return True
