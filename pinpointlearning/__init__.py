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
    mpc._colors_full_map[cname] = cdef


mpl.rcParams["axes.prop_cycle"] = mpl.cycler(
    color=["hclightblue", "hccoral", "hcdarkblue", "hcgrey"]
)

cmap = mpl.colors.LinearSegmentedColormap.from_list(
    "hccmap", ["hclightblue", "#ffffff", "hccoral"]
)

mpl.colormaps.register(cmap, name="hccmap")
