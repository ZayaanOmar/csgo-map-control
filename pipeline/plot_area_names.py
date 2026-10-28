"""
plot_area_names.py
------------------
Colour every de_mirage nav tile by its awpy ``areaName`` on the radar image,
with one unique colour per name (the blank "" name included) and a key.

Outputs (default directory: outputs/area_name_map/):
    de_mirage_area_names.png   - raster, for quick viewing
    de_mirage_area_names.pdf   - vector, for the write-up
    de_mirage_area_names.csv   - areaName, n_nodes, total footprint, node IDs
                                 (answers "which nodes belong to which area")

Usage (from project root):
    python pipeline/plot_area_names.py
    python pipeline/plot_area_names.py --labels            # also write names on the map
    python pipeline/plot_area_names.py --light             # light radar instead of dark
    python pipeline/plot_area_names.py --assignment data/de_mirage_area_assignment.csv
        # use the output of area_assignment.py: no blank group (assigned labels
        # are plotted like any other name; provenance is tracked in the CSV only)
    python pipeline/plot_area_names.py --out-dir some/dir

Notes
-----
* Reads from ``awpy.data.NAV`` (not the patched graph): the 903 areas and their
  names are identical, since patching only added edges.
* Corner coordinates are transformed individually with
  ``awpy.visualization.plot.position_transform`` and the rectangle is built from
  min/max of the transformed corners (the Y transform is a flip, so
  northWest/southEast cannot be assumed to be top-left/bottom-right).
* Tiles that overlap in XY (e.g. Catwalk over Mid, Underpass/Connector) are
  drawn in ascending mean-z order, so higher surfaces end up on top.
* The blank "" name gets a fixed hot-pink colour AND a hatch pattern, so it is
  distinguishable from every other area even in greyscale.
"""

import argparse
import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")  # no display needed (VM / HPC safe)
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.patches import Patch

from awpy.data import NAV
from awpy.visualization.plot import plot_map, position_transform

MAP_NAME = "de_mirage"
BLANK_COLOUR = "#FF00FF"   # reserved for the blank "" areaName
BLANK_HATCH = "///"
BLANK_LABEL = "(blank)"
FILL_ALPHA = 0.70


# --------------------------------------------------------------------------- #
# Palette
# --------------------------------------------------------------------------- #
def build_palette(names: list[str]) -> dict[str, str]:
    """Assign each non-blank name a colour. Deterministic (sorted-name order).

    Uses matplotlib's tab20 + tab20b + tab20c (60 colours). If there are more
    than 60 names the palette cycles (the key would then be ambiguous, so a
    warning is printed). The blank name is handled separately.
    """
    base = []
    for cmap_name in ("tab20", "tab20b", "tab20c"):
        cmap = plt.get_cmap(cmap_name)
        base.extend(matplotlib.colors.to_hex(cmap(i)) for i in range(cmap.N))
    # Drop anything too close to the reserved blank colour (tab20 has a pink)
    base = [c for c in base if not _is_close(c, BLANK_COLOUR)]

    non_blank = sorted(n for n in names if n != "")
    if len(non_blank) > len(base):
        print(
            f"WARNING: {len(non_blank)} names but only {len(base)} distinct colours; "
            "colours will repeat."
        )
    return {name: base[i % len(base)] for i, name in enumerate(non_blank)}


def _is_close(hex_a: str, hex_b: str, tol: float = 0.25) -> bool:
    a = matplotlib.colors.to_rgb(hex_a)
    b = matplotlib.colors.to_rgb(hex_b)
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5 < tol


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def collect_areas(labels=None):
    """Return (tiles, groups).

    labels : optional {area_id: label} (from area_assignment.py) replacing awpy's names
    tiles  : list of dicts with id, name, transformed rect, mean z, abs footprint
    groups : dict[name] -> list of tile dicts
    """
    tiles = []
    for area_id, area in NAV[MAP_NAME].items():
        x1 = position_transform(MAP_NAME, area["northWestX"], "x")
        x2 = position_transform(MAP_NAME, area["southEastX"], "x")
        y1 = position_transform(MAP_NAME, area["northWestY"], "y")
        y2 = position_transform(MAP_NAME, area["southEastY"], "y")

        # Raw world-space footprint (abs: awpy's own tile area can be negative)
        footprint = abs(
            (area["northWestX"] - area["southEastX"])
            * (area["northWestY"] - area["southEastY"])
        )
        tiles.append(
            {
                "id": int(area_id),
                "name": (labels[int(area_id)] if labels is not None
                         else area["areaName"] if isinstance(area["areaName"], str) else ""),
                "x": min(x1, x2),
                "y": min(y1, y2),
                "w": abs(x2 - x1),
                "h": abs(y2 - y1),
                "z": (area["northWestZ"] + area["southEastZ"]) / 2.0,
                "footprint": footprint,
            }
        )

    groups = defaultdict(list)
    for t in tiles:
        groups[t["name"]].append(t)
    return tiles, groups


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #
def make_plot(tiles, groups, palette, dark: bool, labels: bool):
    fig, ax = plot_map(MAP_NAME, map_type="simpleradar", dark=dark)
    fig.set_size_inches(16, 10)

    # Ascending z so higher surfaces are drawn last (on top)
    for t in sorted(tiles, key=lambda d: d["z"]):
        if t["name"] == "":
            colour, hatch = BLANK_COLOUR, BLANK_HATCH
        else:
            colour, hatch = palette[t["name"]], None
        ax.add_patch(
            patches.Rectangle(
                (t["x"], t["y"]),
                t["w"],
                t["h"],
                facecolor=colour,
                edgecolor="black" if hatch is None else colour,
                linewidth=0.3,
                alpha=FILL_ALPHA,
                hatch=hatch,
            )
        )

    # Optional on-map labels at the footprint-weighted centre of each name.
    # NB: for areas made of non-contiguous pieces the centre can fall outside
    # the area itself - treat as an approximate marker, the key is authoritative.
    if labels:
        for name, members in groups.items():
            total = sum(m["footprint"] for m in members) or 1.0
            cx = sum((m["x"] + m["w"] / 2) * m["footprint"] for m in members) / total
            cy = sum((m["y"] + m["h"] / 2) * m["footprint"] for m in members) / total
            ax.text(
                cx,
                cy,
                name if name else BLANK_LABEL,
                fontsize=6,
                ha="center",
                va="center",
                color="white" if dark else "black",
                bbox=dict(
                    boxstyle="round,pad=0.15",
                    facecolor="black" if dark else "white",
                    alpha=0.6,
                    linewidth=0,
                ),
            )

    # Key: alphabetical, blank first, with node counts
    ordered = [""] + sorted(n for n in groups if n != "") if "" in groups else sorted(groups)
    handles = []
    for name in ordered:
        n = len(groups[name])
        if name == "":
            handles.append(
                Patch(
                    facecolor=BLANK_COLOUR,
                    edgecolor=BLANK_COLOUR,
                    hatch=BLANK_HATCH,
                    alpha=FILL_ALPHA,
                    label=f"{BLANK_LABEL} ({n})",
                )
            )
        else:
            handles.append(
                Patch(
                    facecolor=palette[name],
                    edgecolor="black",
                    linewidth=0.3,
                    alpha=FILL_ALPHA,
                    label=f"{name} ({n})",
                )
            )

    ncol = 1 if len(handles) <= 28 else 2
    leg = ax.legend(
        handles=handles,
        title="areaName (number of nav nodes)",
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=8,
        title_fontsize=9,
        ncol=ncol,
        frameon=False,
    )
    leg._legend_box.align = "left"

    ax.set_title(
        f"{MAP_NAME}: nav mesh coloured by area name "
        f"({len(tiles)} nodes, {len(groups)} area names)",
        fontsize=12,
    )
    ax.axis("off")
    return fig


# --------------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------------- #
def write_csv(groups, path: str):
    with open(path, "w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(["areaName", "n_nodes", "total_footprint", "mean_z", "node_ids"])
        for name in sorted(groups, key=lambda n: (n != "", n)):
            members = groups[name]
            w.writerow(
                [
                    name if name else BLANK_LABEL,
                    len(members),
                    round(sum(m["footprint"] for m in members), 1),
                    round(sum(m["z"] for m in members) / len(members), 1),
                    " ".join(str(m["id"]) for m in sorted(members, key=lambda m: m["id"])),
                ]
            )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="outputs/area_name_map")
    ap.add_argument("--light", action="store_true", help="use the light radar (default: dark)")
    ap.add_argument("--labels", action="store_true", help="also write area names on the map")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--assignment", default=None,
                    help="CSV from area_assignment.py; replaces blank names with assigned labels")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    labels = None
    if args.assignment:
        from area_assignment import load_assignment   # same directory
        labels, _ = load_assignment(args.assignment)
    tiles, groups = collect_areas(labels)
    palette = build_palette(list(groups))

    print(f"{len(tiles)} nodes, {len(groups)} distinct areaNames "
          f"(blank: {len(groups.get('', []))} nodes)")
    for name in sorted(groups, key=lambda n: (n != "", n)):
        print(f"  {name if name else BLANK_LABEL:<24} {len(groups[name]):>4} nodes")

    fig = make_plot(tiles, groups, palette, dark=not args.light, labels=args.labels)

    stem = os.path.join(args.out_dir,
                        "de_mirage_area_names" + ("_assigned" if args.assignment else ""))
    fig.savefig(stem + ".png", dpi=args.dpi, bbox_inches="tight")
    fig.savefig(stem + ".pdf", bbox_inches="tight")
    plt.close(fig)

    write_csv(groups, stem + ".csv")
    print(f"\nWrote:\n  {stem}.png\n  {stem}.pdf\n  {stem}.csv")


if __name__ == "__main__":
    main()
