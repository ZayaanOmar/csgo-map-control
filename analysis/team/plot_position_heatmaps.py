#!/usr/bin/env python
"""
Heatmaps of average player positions at time t after freeze end, for T and CT, on full-buy vs
eco rounds, drawn over the de_mirage radar (awpy's own radar image and coordinate transform).

What is plotted (one PNG per side x condition = 4 PNGs per time t)
  * For every selected round, the frame nearest to t seconds after freeze end is taken (frames
    are ~0.5 s apart; rounds with no frame within --tol seconds of t are dropped, e.g. rounds
    that ended earlier). All ALIVE players of the side at that frame are plotted.
  * Density = mean number of alive players per round per 100x100-game-unit cell, smoothed with
    a Gaussian (--bandwidth game units). Each round counts once, so eco and full-buy panels with
    different round counts are comparable. The eco and full-buy PNGs of a side share one colour
    scale, so they can be compared side by side.

Conditions (non-pistol, non-overtime rounds), from the side's own point of view:
  eco  = own buy class in --eco-classes  (default: eco = "Full Eco")
  full = own buy class in --full-classes (default: full = "Full Buy")
  both against an opponent in --opponent (default: "full" = full-buy opponent; "any" = no condition)
By default only team-matches (team + side + match) that contain BOTH conditions are used, as in
analysis_eco_vs_full.py, so the comparison is within team and match. --all-rounds disables that.

Inputs : outputs/position_maps/*.csv, outputs/round_metadata/*.csv, outputs/analysis/round_economy.csv
Output : outputs/analysis/figures/position_heatmaps/{T|CT}_{eco|full}_t{t}s.png

Geometry (--geometry auto, the default: "diffuse" if --apsp exists, otherwise "mask"):
  diffuse : smooth like a Gaussian blur, but as HEAT DIFFUSION over the walkable area, so it follows
          corridors and cannot cross walls. The nav-mesh tiles are rasterised at 10-unit cells;
          every player is dropped at their real position (clamped into their own nav tile); heat
          then flows only between neighbouring walkable cells (no flux into walls or off the map),
          for the number of steps that corresponds to a Gaussian of width --bandwidth. Flow
          between two touching tiles is additionally blocked when the walking distance between
          them exceeds --cut x the straight-line distance + 40 units (tiles that touch on the
          radar but are not connected in the nav graph, e.g. different levels or across a ledge).
          Mass is conserved: the map integrates to the mean number of alive players per round.
          --cut 0 disables the graph test and keeps only the raster walls.
  graph : each player's mass is spread over the NAV GRAPH, not the plane. A player on tile i adds
          K(d_ij) / sum_k(area_k K(d_ik)) to tile j, with d the walking (geodesic) distance from
          data/de_mirage_apsp_dict.pkl (symmetrised: min of the two directions; 33% of finite pairs
          are asymmetric), K a Gaussian of width --bandwidth, and area_k the tile footprint.
          Every player integrates to exactly one, so the map integrates to the mean number of
          alive players per round. Mass cannot cross a wall, because tiles on either side of it
          are far apart by walking; tiles are painted flat, so resolution is one nav tile
          (median footprint about 43 x 43 units) and the picture conforms exactly to the walkable
          geometry. Densities are per unit of WALKABLE area, so narrow corridors read higher than the
          same players in open ground; do not compare colours with mask/none maps.
  mask  : plain Gaussian smoothing spreads mass over walls and out
of the map. With the mask, the nav-mesh tiles are rasterised onto the radar to get the walkable
area (padded by --mask-pad pixels, 5 game units each); smoothing is then boundary-corrected
(smoothed counts divided by the smoothed walkable fraction, so mass near a wall is kept inside
instead of leaking out) and everything outside the walkable area is not drawn. A thin wall
between two walkable corridors is still bridged by the blur: lower --bandwidth to reduce that.
--geometry none gives the old unmasked maps. --labels writes area names on the radar so hotspots
can be read directly.

Choosing t: frames are ~0.5 s apart, so t snaps to the nearest frame within --tol (0.3 s). Data
start at freeze end (t >= 0) and each round ends at its own time (median ~85 s). A round that
ended before t cannot contribute, so at late t the panels describe only the rounds that lasted;
the console and each figure's subtitle report "x of y rounds lasted to t", and a note is printed
when eco and full-buy rounds survive at clearly different rates (the two panels then describe
different kinds of rounds). A t outside the data stops with the valid range instead of writing
blank figures.

Per-team mode: --team "G2 Esports" draws the same four figures from that team's rounds only
(repeat --team for several teams). Names are the canonical names in data/team_aliases.json; a
near-miss prints suggestions. --list-teams shows each team's eco / full-buy round counts so you
can pick one with enough data (a team with few eco rounds gives a noisy map; the subtitle of
every figure shows the round count). In team mode the default is ALL of the team's rounds (one
team already controls for team style); add --paired to keep only matches with both conditions.

Usage  : python3 plot_position_heatmaps.py                       # all teams pooled, t = 10 s
         python3 plot_position_heatmaps.py --times 5 10 15 20
         python3 plot_position_heatmaps.py --team "G2 Esports" --times 10
         python3 plot_position_heatmaps.py --team "FaZe Clan" --team "Natus Vincere"
         python3 plot_position_heatmaps.py --list-teams
"""
import argparse
import glob
import difflib
import os
import pickle
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from awpy.data import MAP_DATA, NAV
from awpy.visualization.plot import plot_map, position_transform
from scipy import sparse
from scipy.ndimage import binary_dilation, gaussian_filter, zoom

MAP_NAME = "de_mirage"
TICK_RATE = 128.0            # demos report 127 or 128; < 1% difference
PM_GLOB = "outputs/position_maps/*.csv"
RM_GLOB = "outputs/round_metadata/*.csv"
ECON_PATH = "outputs/analysis/round_economy.csv"
APSP_PATH = "data/de_mirage_apsp_dict.pkl"
DEFAULT_OUT_DIR = "outputs/analysis/figures/position_heatmaps"
# floor (walkable-area) colour of awpy's simple radar, measured from the images (23% of the pixels)
RADAR_FLOOR = {"dark": (1.0 / 255.0, 19.0 / 255.0, 48.0 / 255.0), "light": (38.0 / 255.0, 50.0 / 255.0, 70.0 / 255.0)}
MIN_ROUNDS_WARN = 15       # below this many rounds a panel is flagged as noisy
DIFF_RES = 2                 # radar pixels per cell of the wall-aware diffusion grid (10 game units)
ALPHA = 0.2                  # diffusion step weight (4-neighbour, stable below 0.25)
BIN_PX = 4                   # histogram bin in radar pixels (5 game units per pixel on de_mirage)
SIDES = ("T", "CT")
ALL_CLASSES = ["eco", "semi_eco", "semi_buy", "full"]


# ------------------------------------------------------------------ data
def condition_table(econ, eco_classes, full_classes, opp_classes):
    e = econ[~econ["is_pistol"] & ~econ["is_overtime"]]
    parts = []
    for side, own, opp, team in (("T", "t_buy", "ct_buy", "t_team"), ("CT", "ct_buy", "t_buy", "ct_team")):
        d = e[["match_id", "match_name", "round", own, opp, team]].rename(columns={own: "own", opp: "opp", team: "team"})
        d = d[d["opp"].isin(opp_classes)]
        d["cond"] = np.where(d["own"].isin(eco_classes), "eco", np.where(d["own"].isin(full_classes), "full", "other"))
        d = d[d["cond"] != "other"].assign(side=side)
        parts.append(d[["match_id", "match_name", "round", "side", "team", "cond"]])
    return pd.concat(parts, ignore_index=True)


def collect_positions(times, tol, names=None):
    """For each match and requested time, the alive-player positions at the frame nearest to t."""
    rm = pd.concat((pd.read_csv(p, dtype={"match_id": str}) for p in glob.glob(RM_GLOB)), ignore_index=True)
    rm = rm[["match_id", "round", "freeze_time_end_tick"]]
    pts, rounds, cov, ranges = [], [], [], []
    files = sorted(glob.glob(PM_GLOB))
    if names is not None:
        files = [p for p in files if os.path.basename(p)[: -len(".csv")] in names]
        assert files, "no position_maps files match the selected teams' matches"
    print(f"reading {len(files)} position_maps files")
    for i, p in enumerate(files, 1):
        pm = pd.read_csv(p, usecols=["match_id", "round", "frame_idx", "side", "tick", "current_x", "current_y", "current_areaId"],
                         dtype={"match_id": str})
        mid = pm["match_id"].iloc[0]
        pm = pm.merge(rm[rm["match_id"] == mid][["round", "freeze_time_end_tick"]], on="round", how="left")
        assert pm["freeze_time_end_tick"].notna().all(), f"{os.path.basename(p)}: rounds missing from round_metadata"
        pm["sec"] = (pm["tick"] - pm["freeze_time_end_tick"]) / TICK_RATE
        fr = pm.drop_duplicates(["round", "frame_idx"])[["round", "frame_idx", "sec"]]
        bounds = fr.groupby("round")["sec"].agg(first="min", last="max")
        ranges.append(bounds.reset_index().assign(match_id=mid))
        for t in times:
            gap = (fr["sec"] - t).abs()
            best = fr.loc[gap.groupby(fr["round"]).idxmin()]
            best = best[(best["sec"] - t).abs() <= tol].assign(t=t)[["t", "round", "frame_idx"]]
            rounds.append(best[["t", "round"]].assign(match_id=mid))
            status = np.where(bounds.index.isin(best["round"]), "ok",
                              np.where(t > bounds["last"].to_numpy(), "ended", "early"))
            cov.append(pd.DataFrame({"round": bounds.index, "status": status}).assign(t=t, match_id=mid))
            pts.append(pm.merge(best, on=["round", "frame_idx"])
                         [["t", "match_id", "round", "side", "current_x", "current_y", "current_areaId"]])
        if i % 50 == 0 or i == len(files):
            print(f"  read {i}/{len(files)} position files")
    return (pd.concat(pts, ignore_index=True), pd.concat(rounds, ignore_index=True),
            pd.concat(cov, ignore_index=True), pd.concat(ranges, ignore_index=True))


# ------------------------------------------------------------------ density + plotting
def walkable_mask(shape_hw, pad_px):
    """Boolean radar-pixel raster of the union of nav-mesh tiles, dilated by pad_px pixels."""
    h_px, w_px = shape_hw
    m = np.zeros(shape_hw, dtype=bool)
    for a in NAV[MAP_NAME].values():
        xs = position_transform(MAP_NAME, np.array([a["northWestX"], a["southEastX"]], float), "x")
        ys = position_transform(MAP_NAME, np.array([a["northWestY"], a["southEastY"]], float), "y")
        x0, x1 = int(np.floor(min(xs))), int(np.ceil(max(xs)))
        y0, y1 = int(np.floor(min(ys))), int(np.ceil(max(ys)))
        m[max(y0, 0):min(y1, h_px), max(x0, 0):min(x1, w_px)] = True
    return binary_dilation(m, iterations=pad_px) if pad_px > 0 else m


def area_label_positions():
    """Radar-pixel centroid of each named area (mean of its nav tiles)."""
    pos = {}
    for a in NAV[MAP_NAME].values():
        xs = position_transform(MAP_NAME, np.array([a["northWestX"], a["southEastX"]], float), "x")
        ys = position_transform(MAP_NAME, np.array([a["northWestY"], a["southEastY"]], float), "y")
        pos.setdefault(a["areaName"], []).append((xs.mean(), ys.mean()))
    return {k: tuple(np.mean(v, axis=0)) for k, v in pos.items()}


def load_graph_kernel(apsp_path, bandwidth):
    """Graph-diffusion kernel W (nodes x nodes) in 'players per unit area per player', plus tile areas.
    W[i, j] = K(d_ij) / sum_k area_k K(d_ik), so sum_j area_j W[i, j] = 1 for every source node i."""
    with open(apsp_path, "rb") as f:
        apsp = pickle.load(f)
    D = np.asarray(apsp["distance_matrix"], dtype=float)
    a2i = apsp["area_to_index_map"]
    ids = sorted(a2i, key=a2i.get)                      # area ids in matrix order
    nav = NAV[MAP_NAME]
    missing = [a for a in ids if a not in nav]
    assert not missing, f"{len(missing)} APSP areas are not in awpy's NAV for {MAP_NAME}"
    area = np.array([abs((nav[a]["southEastX"] - nav[a]["northWestX"]) * (nav[a]["southEastY"] - nav[a]["northWestY"]))
                     for a in ids], dtype=float)
    area = np.maximum(area, 1.0)
    Ds = np.minimum(D, D.T)
    finite = np.isfinite(Ds)
    K = np.where(finite, np.exp(-(np.where(finite, Ds, 0.0) ** 2) / (2.0 * bandwidth ** 2)), 0.0)
    W = K / (K @ area)[:, None]
    return W, a2i, ids, area


def node_raster(ids, area, shape_hw):
    """Radar-pixel raster whose value is the matrix index of the nav tile covering that pixel (-1 = none)."""
    h_px, w_px = shape_hw
    ras = np.full(shape_hw, -1, dtype=np.int32)
    nav = NAV[MAP_NAME]
    for k in np.argsort(-area):                          # big tiles first, small ones paint over them
        a = nav[ids[k]]
        xs = position_transform(MAP_NAME, np.array([a["northWestX"], a["southEastX"]], float), "x")
        ys = position_transform(MAP_NAME, np.array([a["northWestY"], a["southEastY"]], float), "y")
        ras[max(int(np.floor(min(ys))), 0):min(int(np.ceil(max(ys))), h_px),
            max(int(np.floor(min(xs))), 0):min(int(np.ceil(max(xs))), w_px)] = k
    return ras


def graph_density(node_idx, n_rounds, W):
    """Per-round players per 100x100 units at every tile: (player-rounds per node / rounds) @ W."""
    c = np.bincount(node_idx, minlength=W.shape[0]) / max(n_rounds, 1)
    return (c @ W) * 1e4


def build_diffusion(apsp_path, bandwidth, cut, shape_hw, scale):
    """Sparse wall-aware diffusion operator on the walkable cells of the radar (DIFF_RES px per cell)."""
    with open(apsp_path, "rb") as f:
        apsp = pickle.load(f)
    D = np.asarray(apsp["distance_matrix"], dtype=float)
    a2i = apsp["area_to_index_map"]
    ids = sorted(a2i, key=a2i.get)
    nav = NAV[MAP_NAME]
    assert all(a in nav for a in ids), "APSP areas missing from awpy NAV"
    n = len(ids)
    area = np.array([abs((nav[a]["southEastX"] - nav[a]["northWestX"]) * (nav[a]["southEastY"] - nav[a]["northWestY"]))
                     for a in ids], dtype=float)
    cx = np.array([(nav[a]["northWestX"] + nav[a]["southEastX"]) / 2 for a in ids])
    cy = np.array([(nav[a]["northWestY"] + nav[a]["southEastY"]) / 2 for a in ids])
    Ds = np.minimum(D, D.T)
    eu = np.hypot(cx[:, None] - cx[None, :], cy[:, None] - cy[None, :])
    connected = np.ones((n, n), dtype=bool) if cut <= 0 else (np.isfinite(Ds) & (Ds <= cut * eu + 40.0))
    np.fill_diagonal(connected, True)

    h_px, w_px = shape_hw
    assert h_px % DIFF_RES == 0 and w_px % DIFF_RES == 0, "radar size must be divisible by DIFF_RES"
    hc, wc = h_px // DIFF_RES, w_px // DIFF_RES
    ras = np.full((hc, wc), -1, dtype=np.int32)
    boxes = np.zeros((n, 4), dtype=int)                  # fallback x0, x1, y0, y1 (cell coords) for tiles that own no cell
    for k in np.argsort(-area):
        a = nav[ids[k]]
        xs = position_transform(MAP_NAME, np.array([a["northWestX"], a["southEastX"]], float), "x") / DIFF_RES
        ys = position_transform(MAP_NAME, np.array([a["northWestY"], a["southEastY"]], float), "y") / DIFF_RES
        # a cell belongs to the tile its CENTRE falls in, so abutting tiles share no cells; a tile smaller
        # than one cell still gets the cell that contains its centre
        x0, x1 = int(np.ceil(xs.min() - 0.5)), int(np.ceil(xs.max() - 0.5))
        y0, y1 = int(np.ceil(ys.min() - 0.5)), int(np.ceil(ys.max() - 0.5))
        if x1 <= x0:
            x0 = int(np.floor(xs.mean())); x1 = x0 + 1
        if y1 <= y0:
            y0 = int(np.floor(ys.mean())); y1 = y0 + 1
        x0, x1, y0, y1 = max(x0, 0), min(x1, wc), max(y0, 0), min(y1, hc)
        if x1 > x0 and y1 > y0:
            ras[y0:y1, x0:x1] = k
        boxes[k] = (x0, x1, y0, y1)
    flat = ras.ravel()
    owned_order = np.argsort(flat, kind="stable")
    owned_count = np.bincount(flat[flat >= 0], minlength=n)
    owned_start = np.concatenate([[0], np.cumsum(owned_count)[:-1]]) + int((flat < 0).sum())
    walk = ras >= 0
    cid = np.full((hc, wc), -1, dtype=np.int64)
    cid[walk] = np.arange(int(walk.sum()))

    rows, cols, n_touch, n_cut = [], [], 0, 0
    for dy, dx in ((0, 1), (1, 0)):
        a, b = ras[:hc - dy, :wc - dx], ras[dy:, dx:]
        both = (a >= 0) & (b >= 0)
        open_ = both & connected[np.clip(a, 0, None), np.clip(b, 0, None)]
        n_touch += int((both & (a != b)).sum())
        n_cut += int((both & (a != b) & ~open_).sum())
        ia, ib = cid[:hc - dy, :wc - dx][open_], cid[dy:, dx:][open_]
        rows += [ia, ib]
        cols += [ib, ia]
    N = int(walk.sum())
    S = sparse.csr_matrix((np.ones(sum(len(r) for r in rows)), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N))
    deg = np.asarray(S.sum(axis=1)).ravel()
    P = sparse.identity(N, format="csr") - ALPHA * sparse.diags(deg) + ALPHA * S
    sigma_cells = bandwidth / (scale * DIFF_RES)
    steps = int(np.ceil(sigma_cells ** 2 / (2.0 * ALPHA)))
    return {"P": P.tocsr(), "steps": steps, "cid": cid, "walk": walk, "boxes": boxes, "a2i": a2i, "ras_c": ras,
            "owned_order": owned_order, "owned_start": owned_start, "owned_count": owned_count,
            "px_ras": node_raster(ids, area, shape_hw), "scale": scale, "n_cells": N,
            "cut_share": n_cut / max(n_touch, 1), "n_touch": n_touch}


def inject_cells(model, node_idx, x_px, y_px):
    """Cell index for every player: the nearest cell OWNED by the player's own nav tile (falls back to the
    tile's box when other tiles cover all of it, which only happens where levels stack on the radar)."""
    hc, wc = model["walk"].shape
    xc, yc = x_px / DIFF_RES, y_px / DIFF_RES
    out = np.full(len(node_idx), -1, dtype=np.int64)
    for k in np.unique(node_idx):
        sel = np.flatnonzero(node_idx == k)
        cnt = model["owned_count"][k]
        if cnt:
            flat = model["owned_order"][model["owned_start"][k]: model["owned_start"][k] + cnt]
            oy, ox = flat // wc, flat % wc
            d2 = (xc[sel, None] - (ox[None, :] + 0.5)) ** 2 + (yc[sel, None] - (oy[None, :] + 0.5)) ** 2
            j = d2.argmin(axis=1)
            out[sel] = model["cid"][oy[j], ox[j]]
        else:
            x0, x1, y0, y1 = model["boxes"][k]
            cx_ = np.clip(np.floor(xc[sel]).astype(int), x0, max(x1 - 1, x0))
            cy_ = np.clip(np.floor(yc[sel]).astype(int), y0, max(y1 - 1, y0))
            out[sel] = model["cid"][np.clip(cy_, 0, hc - 1), np.clip(cx_, 0, wc - 1)]
    return out


def diffuse_density(model, node_idx, x_px, y_px, n_rounds):
    """Players per round per 100x100 units on every walkable cell after wall-aware diffusion."""
    cells = inject_cells(model, np.asarray(node_idx), np.asarray(x_px, float), np.asarray(y_px, float))
    u = np.bincount(cells[cells >= 0], minlength=model["n_cells"]).astype(float) / max(n_rounds, 1)
    P = model["P"]
    for _ in range(model["steps"]):
        u = P @ u
    return u * (1e4 / (DIFF_RES * model["scale"]) ** 2)


def cells_to_pixels(model, u):
    """Upsample cell densities to radar pixels (normalised by the walkable fraction) and blank non-walkable pixels."""
    hc, wc = model["walk"].shape
    g = np.zeros((hc, wc))
    g[model["walk"]] = u
    up = zoom(g, DIFF_RES, order=1, grid_mode=True, mode="grid-constant")
    um = zoom(model["walk"].astype(float), DIFF_RES, order=1, grid_mode=True, mode="grid-constant")
    out = up / np.maximum(um, 0.2)
    out[model["px_ras"] < 0] = np.nan
    return out


def density(xs, ys, n_rounds, shape_hw):
    """Mean alive players per round per 100x100-unit cell, Gaussian-smoothed (sigma set by caller)."""
    h_px, w_px = shape_hw
    xe, ye = np.arange(0, w_px + BIN_PX, BIN_PX), np.arange(0, h_px + BIN_PX, BIN_PX)
    h, _, _ = np.histogram2d(xs, ys, bins=[xe, ye])
    return h.T / max(n_rounds, 1)           # rows = y, columns = x; per-bin, scaled by the caller


def smooth_and_scale(h, sigma_bins, scale_units_per_px, mask_frac=None):
    """Gaussian-smoothed per-round density in players per 100x100 units. With mask_frac (walkable
    fraction of each bin) the smoothing is boundary-corrected: smoothed counts / smoothed mask."""
    cell = (BIN_PX * scale_units_per_px) ** 2
    num = gaussian_filter(h, sigma_bins, mode="constant")
    if mask_frac is not None:
        num = num / np.maximum(gaussian_filter(mask_frac, sigma_bins, mode="constant"), 0.15)
    return num * (100.0 ** 2 / cell)


def resolve_team(name, known):
    """Exact match, then case/punctuation-insensitive match; otherwise exit with suggestions."""
    if name in known:
        return name
    norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
    hits = [k for k in known if norm(k) == norm(name)]
    if len(hits) == 1:
        return hits[0]
    close = difflib.get_close_matches(name, known, n=5, cutoff=0.4)
    raise SystemExit(f"team {name!r} not found. Closest names: {close or 'none'}. "
                     f"Run with --list-teams to see all names and their round counts.")


def slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")


def heat_cmap(floor_rgb, blend=0.12):
    """Inferno whose 0.0 end is the radar floor colour, so zero density is indistinguishable from the map
    and the ramp fades into inferno over the first `blend` of the scale."""
    base = plt.get_cmap("inferno")(np.linspace(0.0, 1.0, 256))
    w = np.clip(1.0 - np.linspace(0.0, 1.0, 256) / blend, 0.0, 1.0)[:, None]
    base[:, :3] = w * np.asarray(floor_rgb, float) + (1.0 - w) * base[:, :3]
    return ListedColormap(base, name="inferno_on_radar")


def save_panel(path, bg, dens, vmax, title, subtitle, mask_px=None, labels=None, pixel=False, cmap=None):
    """One square PNG: radar + density overlay + colour bar (shared vmax within a side)."""
    h_px, w_px = bg.shape[:2]
    fig, ax = plt.subplots(figsize=(7.4, 7.4))
    ax.imshow(bg)
    if dens is None:
        ax.text(0.5, 0.5, "no rounds", transform=ax.transAxes, ha="center", color="w")
    else:
        if pixel:                           # graph mode: dens is already a radar-pixel raster (NaN off the mesh)
            shown = np.ma.masked_where(np.isnan(dens) | (dens < 0.04 * vmax), dens)
            im = ax.imshow(shown, extent=(0, w_px, h_px, 0), cmap=cmap or "inferno", alpha=0.9,
                           vmin=0, vmax=vmax, interpolation="nearest")
        elif mask_px is not None:           # pixel-resolution, clipped to the walkable area
            d_px = zoom(dens, BIN_PX, order=1)
            shown = np.ma.masked_where(~mask_px | (d_px < 0.04 * vmax), d_px)
            im = ax.imshow(shown, extent=(0, w_px, h_px, 0), cmap=cmap or "inferno", alpha=0.9,
                           vmin=0, vmax=vmax, interpolation="nearest")
        else:
            shown = np.ma.masked_less(dens, 0.04 * vmax)
            im = ax.imshow(shown, extent=(0, w_px, h_px, 0), cmap=cmap or "inferno", alpha=0.85,
                           vmin=0, vmax=vmax, interpolation="bilinear")
        cb = fig.colorbar(im, ax=ax, shrink=0.78, pad=0.015)
        cb.set_label("alive players per round per 100x100 units", fontsize=8)
        try:                                   # draw the colour bar opaque so its 0.0 end is exactly the radar floor colour
            cb.solids.set_alpha(1.0)
        except AttributeError:
            pass
    if labels:
        for name, (lx, ly) in labels.items():
            ax.text(lx, ly, name, fontsize=5.5, color="w", ha="center", va="center", alpha=0.9,
                    bbox=dict(boxstyle="round,pad=0.12", fc="k", ec="none", alpha=0.35))
    ax.set_title(title, fontsize=11, pad=26)
    ax.text(0.5, 1.012, subtitle, transform=ax.transAxes, ha="center", va="bottom", fontsize=7.5, color="0.35")
    ax.set_axis_off()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--times", type=float, nargs="+", default=[10.0], help="seconds after freeze end")
    ap.add_argument("--tol", type=float, default=0.3, help="max |frame time - t| in seconds")
    ap.add_argument("--opponent", choices=("full", "any"), default="full")
    ap.add_argument("--eco-classes", nargs="+", default=["eco"], choices=ALL_CLASSES)
    ap.add_argument("--full-classes", nargs="+", default=["full"], choices=ALL_CLASSES)
    ap.add_argument("--team", action="append", help='canonical team name, e.g. --team "G2 Esports" (repeatable)')
    ap.add_argument("--list-teams", action="store_true", help="print each team's eco/full round counts and exit")
    ap.add_argument("--paired", action="store_true", help="only team-matches containing both conditions (default for pooled mode)")
    ap.add_argument("--all-rounds", action="store_true", help="do not restrict to team-matches with both conditions (default in team mode)")
    ap.add_argument("--bandwidth", type=float, default=100.0, help="Gaussian smoothing in game units")
    ap.add_argument("--geometry", choices=("auto", "diffuse", "graph", "mask", "none"), default="auto",
                    help="diffuse: smooth, wall-aware heat diffusion on the nav mesh; graph: flat per-tile geodesic kernel; "
                         "mask: Gaussian clipped to the walkable area; none: plain Gaussian; "
                         "auto (default): diffuse if --apsp exists, else mask")
    ap.add_argument("--cut", type=float, default=1.6,
                    help="diffuse mode: block flow between touching tiles whose walking distance exceeds "
                         "CUT x straight-line + 40 units (0 = only raster walls)")
    ap.add_argument("--apsp", default=APSP_PATH, help="pickle with distance_matrix / area_to_index_map")
    ap.add_argument("--mask-pad", type=int, default=3, help="walkable-mask padding in radar pixels (5 units each)")
    ap.add_argument("--labels", action="store_true", help="write area names on the radar")
    ap.add_argument("--light", action="store_true", help="light radar background instead of dark")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = ap.parse_args()
    assert not set(args.eco_classes) & set(args.full_classes), "eco and full classes overlap"
    if any(t < 0 for t in args.times):
        raise SystemExit("--times must be >= 0: frames start at freeze end, so there is no data before 0 s")
    os.makedirs(args.out_dir, exist_ok=True)

    fig0, ax0 = plot_map(MAP_NAME, "simpleradar", dark=not args.light)
    bg = np.asarray(ax0.images[0].get_array())
    plt.close(fig0)
    scale = float(MAP_DATA[MAP_NAME]["scale"])
    sigma_bins = args.bandwidth / (scale * BIN_PX)
    geometry = args.geometry
    if geometry == "auto":
        geometry = "diffuse" if os.path.exists(args.apsp) else "mask"
        print(f"--geometry auto -> {geometry}" + ("" if geometry == "diffuse" else f" ({args.apsp} not found)"))
    elif geometry in ("graph", "diffuse") and not os.path.exists(args.apsp):
        raise SystemExit(f"--geometry {geometry} needs {args.apsp} (use --apsp to point at it)")
    W = a2i = node_ras = model = None
    if geometry == "diffuse":
        model = build_diffusion(args.apsp, args.bandwidth, args.cut, bg.shape[:2], float(MAP_DATA[MAP_NAME]["scale"]))
        a2i = model["a2i"]
        print(f"diffusion grid: {model['n_cells']} walkable cells of {DIFF_RES * float(MAP_DATA[MAP_NAME]['scale']):g} units, "
              f"{model['steps']} steps for sigma {args.bandwidth:g} units; {model['cut_share']:.1%} of "
              f"{model['n_touch']} touching tile boundaries are blocked by the graph test (--cut {args.cut:g})")
    if geometry == "graph":
        W, a2i, ids, tile_area = load_graph_kernel(args.apsp, args.bandwidth)
        node_ras = node_raster(ids, tile_area, bg.shape[:2])
        print(f"graph kernel: {W.shape[0]} nodes, sigma {args.bandwidth:g} units along the nav graph "
              f"(median tile footprint {np.median(tile_area) ** 0.5:.0f} x {np.median(tile_area) ** 0.5:.0f} units)")
    smooth_desc = {"diffuse": f"wall-aware diffusion on the nav mesh, sigma {args.bandwidth:g} u",
                   "graph": f"nav-graph kernel, sigma {args.bandwidth:g} u (tile resolution)",
                   "mask": f"Gaussian clipped to walkable area, sigma {args.bandwidth:g} u",
                   "none": f"plain Gaussian, sigma {args.bandwidth:g} u"}[geometry]
    mask_px = mask_frac = None
    if geometry == "mask":
        assert bg.shape[0] % BIN_PX == 0 and bg.shape[1] % BIN_PX == 0, "radar size must be divisible by BIN_PX"
        mask_px = walkable_mask(bg.shape[:2], args.mask_pad)
        mask_frac = mask_px.reshape(bg.shape[0] // BIN_PX, BIN_PX, bg.shape[1] // BIN_PX, BIN_PX).mean(axis=(1, 3))
        print(f"walkable mask: {mask_px.mean():.1%} of the radar (nav tiles padded {args.mask_pad} px)")
    label_pos = area_label_positions() if args.labels else None
    print(f"radar {bg.shape[1]}x{bg.shape[0]} px, {scale:g} units/px, smoothing sigma {args.bandwidth:g} units")

    econ = pd.read_csv(ECON_PATH, dtype={"match_id": str})
    opp = ["full"] if args.opponent == "full" else ALL_CLASSES
    conds = condition_table(econ, args.eco_classes, args.full_classes, opp)
    print(f"{len(conds)} candidate team-rounds (eco classes {args.eco_classes}, full classes "
          f"{args.full_classes}, opponent {args.opponent})")
    if args.list_teams:
        tab = conds.groupby(["team", "side", "cond"]).size().unstack(["side", "cond"], fill_value=0)
        tab.columns = [f"{s}_{c}" for s, c in tab.columns]
        print(tab.sort_values(tab.columns[0], ascending=False).to_string())
        print("\n(rounds per team, side and condition under the current class/opponent settings)")
        return

    known = sorted(conds["team"].unique())
    targets = [resolve_team(n, known) for n in args.team] if args.team else [None]
    paired = args.paired or (not args.all_rounds and args.team is None)
    names = None if args.team is None else set(conds[conds["team"].isin(targets)]["match_name"])

    pts_all, rounds_all, cov_all, ranges_all = collect_positions(args.times, args.tol, names)
    rng_txt = (f"frames in the data read cover {ranges_all['first'].min():.1f} to {ranges_all['last'].max():.1f} s "
               f"after freeze end (median round length {ranges_all['last'].median():.0f} s)")
    n_written, n_skipped = 0, 0
    pts_all["x_px"] = position_transform(MAP_NAME, pts_all["current_x"].to_numpy(), "x")
    pts_all["y_px"] = position_transform(MAP_NAME, pts_all["current_y"].to_numpy(), "y")
    if geometry in ("graph", "diffuse"):
        pts_all["node"] = pts_all["current_areaId"].map(a2i)
        unmapped = pts_all["node"].isna().mean()
        print(f"{unmapped:.2%} of the {len(pts_all)} plotted player positions have no nav node and are dropped")
        pts_all = pts_all[pts_all["node"].notna()].copy()
        pts_all["node"] = pts_all["node"].astype(int)
    if mask_px is not None:
        xi = np.clip(pts_all["x_px"].to_numpy().astype(int), 0, bg.shape[1] - 1)
        yi = np.clip(pts_all["y_px"].to_numpy().astype(int), 0, bg.shape[0] - 1)
        print(f"{(~mask_px[yi, xi]).mean():.2%} of the {len(pts_all)} plotted player positions fall outside the "
              f"walkable mask (their mass is redistributed to nearby walkable cells)")
    scope = "team-matches containing both conditions" if paired else "all rounds"
    cmap = heat_cmap(RADAR_FLOOR["light" if args.light else "dark"])
    label = {"eco": "eco", "full": "full buy"}

    for target in targets:
        who = "all teams" if target is None else target
        sub_conds = conds if target is None else conds[conds["team"] == target]
        out_dir = args.out_dir if target is None else os.path.join(args.out_dir, slug(target))
        os.makedirs(out_dir, exist_ok=True)
        for t in args.times:
            present = rounds_all[rounds_all["t"] == t][["match_id", "round"]]
            pts_t = pts_all[pts_all["t"] == t]
            print(f"\n--- {who} | t = {t:g} s ---")
            for side in SIDES:
                cand = sub_conds[sub_conds["side"] == side].merge(
                    cov_all[cov_all["t"] == t][["match_id", "round", "status"]], on=["match_id", "round"], how="left")
                cand["status"] = cand["status"].fillna("no data")
                c = sub_conds[sub_conds["side"] == side].merge(present, on=["match_id", "round"])
                if paired:
                    both = c.groupby(["team", "match_id"])["cond"].nunique()
                    c = c.merge(both[both == 2].rename("_k").reset_index()[["team", "match_id"]], on=["team", "match_id"])
                p = pts_t[pts_t["side"] == side].merge(c[["match_id", "round", "cond"]], on=["match_id", "round"])
                panels = {}
                for cond in ("full", "eco"):
                    n, pc = int((c["cond"] == cond).sum()), p[p["cond"] == cond]
                    dens, vals = None, None
                    if n and geometry == "diffuse":
                        u = diffuse_density(model, pc["node"].to_numpy(), pc["x_px"].to_numpy(), pc["y_px"].to_numpy(), n)
                        dens = cells_to_pixels(model, u)
                        vals = u[u > 0]
                    elif n and geometry == "graph":
                        node_d = graph_density(pc["node"].to_numpy(), n, W)
                        dens = np.where(node_ras >= 0, node_d[np.clip(node_ras, 0, None)], np.nan)
                        vals = node_d[node_d > 0]
                    elif n:
                        h = density(pc["x_px"].to_numpy(), pc["y_px"].to_numpy(), n, bg.shape[:2])
                        dens = smooth_and_scale(h, sigma_bins, scale, mask_frac)
                        ok_ = dens > 0
                        vals = dens[ok_ & (mask_frac > 0.25)] if mask_frac is not None else dens[ok_]
                    cc = cand[cand["cond"] == cond]
                    panels[cond] = {"n": n, "alive": len(pc) / max(n, 1), "dens": dens, "vals": vals, "cand": len(cc),
                                    "ok": int((cc["status"] == "ok").sum()), "ended": int((cc["status"] == "ended").sum()),
                                    "early": int((cc["status"] == "early").sum())}
                if panels["eco"]["n"] + panels["full"]["n"] == 0:
                    n_skipped += 1
                    pe, pf = panels["eco"], panels["full"]
                    reach = pe["ok"] + pf["ok"]
                    if paired and reach:
                        why = (f"{reach} candidate rounds lasted to t, but none are in a team-match that still has both "
                               f"conditions at t (paired mode); add --all-rounds to plot them")
                    else:
                        why = (f"{pe['cand'] + pf['cand']} candidate rounds: {pe['ended'] + pf['ended']} ended before t, "
                               f"{pe['early'] + pf['early']} start after t")
                    print(f"  {side}: SKIPPED at t = {t:g} s ({why})")
                    continue
                rates = {k: panels[k]["ok"] / panels[k]["cand"] for k in panels if panels[k]["cand"]}
                if len(rates) == 2 and abs(rates["eco"] - rates["full"]) > 0.10:
                    print(f"  NOTE ({side}, t = {t:g} s): {rates['eco']:.0%} of eco rounds but {rates['full']:.0%} of full-buy "
                          f"rounds lasted to t, so the two panels describe different kinds of rounds")
                vmax = max((np.percentile(d["vals"], 99.5) for d in panels.values()
                            if d["vals"] is not None and d["vals"].size), default=1.0)
                for cond in ("eco", "full"):
                    d = panels[cond]
                    prefix = "" if target is None else f"{slug(target)}_"
                    path = os.path.join(out_dir, f"{prefix}{side}_{cond}_t{t:g}s.png")
                    low = "  LOW N: map is noisy" if 0 < d["n"] < MIN_ROUNDS_WARN else ""
                    head = f"{target}, " if target else ""
                    save_panel(path, bg, d["dens"], vmax, 
                               f"{head}{side} side, {label[cond]}: average positions {t:g} s after freeze end",
                               f"{d['n']} rounds{' (LOW N)' if low else ''} ({d['ok']} of {d['cand']} lasted to t), "
                               f"{d['alive']:.1f} alive players/round"
                               f" | opponent: {args.opponent} buy | {scope}\nsmoothing: {smooth_desc}",
                               mask_px, label_pos, pixel=(geometry in ("graph", "diffuse")), cmap=cmap)
                    n_written += 1
                    print(f"  {side} {cond:4s}: {d['n']:5d} rounds used ({d['ok']}/{d['cand']} lasted to t), "
                          f"{d['alive']:.2f} alive/round -> {path}{low}")


    if n_skipped:
        print(f"\n{n_skipped} side(s) skipped because no round reached the requested time; {rng_txt}.")
    if n_written == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
