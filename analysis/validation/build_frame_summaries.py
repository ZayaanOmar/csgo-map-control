"""
build_frame_summaries.py
========================
Collapses each match's control field (all nodes x frames, ~70 MB CSV) into ONE ROW PER FRAME
so every downstream analysis reads a small parquet instead of re-scanning node-level data.

Output: outputs/frame_summaries/{match}.parquet  — one row per (round, frame_idx)

Core columns
  round, frame_idx, tick, t_rel, tick_rate, ct_alive, t_alive, ct_hp, t_hp, ct_eq, t_eq, ct_util,
  t_util, bomb_planted, bomb_site                         (from frame_state, i.e. straight from the demo)
  ctrl_full            mean control_val over ALL nodes                (CT perspective, 0.5 = even)
  ctrl_active          mean control_val over ACTIVE nodes (ct_mu + t_mu > ACTIVE_THR = exp(-2), i.e. 2 sigma)
                       [the threshold is applied BEFORE the sigmoid]
  ctrl_active_legacy   same with the project's ORIGINAL threshold exp(-4.5) (3 sigma); used only by the
                       Metric 3 reproduction check
  active_thr           the threshold used (stamp column; a cached file whose stamp differs is rebuilt)
  active_frac          fraction of nodes that are active
  contested_all        fraction of ALL nodes with control_val in [0.4, 0.6]   (Metric 1 as I understand it)
  contested_active     fraction of ACTIVE nodes with control_val in [0.4, 0.6]
  contested_strict_<m>, contested_strict_all_<m>
                       ROBUSTLY contested: control in [0.4, 0.6] AND ct_mu >= m AND t_mu >= m, as a fraction
                       of ACTIVE / ALL nodes. m in MASS_THRS (suffixes from vc.mass_tag: 0p25, 0p50, 1p00).
                       A node where both teams are merely faint scores ~0.5 by symmetry; this removes it.
  ct_held_frac_active / t_held_frac_active   fraction of active nodes with control_val >0.5 / <0.5
  site_ctrl_A, site_ctrl_B   mean control_val over bombsite nodes (CT perspective; T control = 1 - this)
  vor_margin           Voronoi baseline margin (ct_area - t_area)/(ct_area + t_area) in [-1, 1]
Per-area-name columns (<Name> is the sanitised areaName, blank -> UNNAMED)
  area_ctrl__<Name>    mean control_val over the area's nodes
  area_cont__<Name>    fraction of the area's nodes that are ACTIVE and contested
  area_contS__<Name>   fraction of the area's nodes that are ROBUSTLY contested (m = STRICT_MIN_MASS)
  occ_ct__<Name>, occ_t__<Name>    number of alive CT / T players standing in the area

Voronoi note: this REPLICATES voronoi_baseline.py as described in the project notes (geodesic
nearest-player hard assignment on the symmetrised distance min(d[u,v], d[v,u]); ties/unreachable
nodes excluded; tile-area weighted) using position_maps' current_areaId. Compare vor_margin against
outputs/voronoi_metrics for one match before relying on it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validation_common as vc  # noqa: E402

STRICT_COLS = ([f"contested_strict_{vc.mass_tag(m)}" for m in vc.MASS_THRS]
               + [f"contested_strict_all_{vc.mass_tag(m)}" for m in vc.MASS_THRS])
CORE_COLS = (["round", "frame_idx", "tick", "t_rel", "tick_rate", "ct_alive", "t_alive", "ct_hp", "t_hp", "ct_eq",
              "t_eq", "ct_util", "t_util", "bomb_planted", "bomb_site", "active_thr", "ctrl_full", "ctrl_active",
              "ctrl_active_legacy", "active_frac", "contested_all", "contested_active", "ct_held_frac_active",
              "t_held_frac_active", "site_ctrl_A", "site_ctrl_B", "vor_margin"] + STRICT_COLS)
FSTATE_KEEP = ["round", "frame_idx", "tick", "t_rel", "tick_rate", "ct_alive", "t_alive", "ct_hp", "t_hp",
               "ct_eq", "t_eq", "ct_util", "t_util", "bomb_planted", "bomb_site"]
KEY = 100000


def build(match: str, ctx: vc.MapContext) -> pd.DataFrame:
    cf = vc.read_resolved(vc.T_CONTROL / f"{match}.csv", vc.CF_SPEC, "control_fields", optional=("control_val",))
    key = cf["round"].to_numpy(np.int64) * KEY + cf["frame_idx"].to_numpy(np.int64)
    uniq, inv = np.unique(key, return_inverse=True)
    nf, N = len(uniq), ctx.n
    ni = ctx.idx_of(cf["area_id"].to_numpy())
    if (ni < 0).any():
        raise ValueError(f"{match}: control_fields contains area IDs absent from the APSP/graph index")
    CT = np.zeros((nf, N), np.float32)
    T = np.zeros((nf, N), np.float32)
    CT[inv, ni] = cf["ct_mu"].to_numpy(np.float32)
    T[inv, ni] = cf["t_mu"].to_numpy(np.float32)
    CTRL = np.full((nf, N), 0.5, np.float32)
    if "control_val" in cf.columns:
        CTRL[inv, ni] = cf["control_val"].to_numpy(np.float32)
    else:
        CTRL[inv, ni] = expit(vc.K_SIGMOID * (cf["ct_mu"].to_numpy(np.float32) - cf["t_mu"].to_numpy(np.float32)))
    del cf
    tot = CT + T
    active = tot > vc.ACTIVE_THR
    active_leg = tot > vc.LEGACY_ACTIVE_THR
    cont = (CTRL >= vc.CONTEST_LO) & (CTRL <= vc.CONTEST_HI)
    n_act, n_leg = active.sum(1), active_leg.sum(1)
    safe = np.maximum(n_act, 1)
    out = {"round": (uniq // KEY).astype(np.int32), "frame_idx": (uniq % KEY).astype(np.int32)}
    out["active_thr"] = np.full(nf, vc.ACTIVE_THR)
    out["ctrl_full"] = CTRL.mean(1)
    out["ctrl_active"] = np.where(n_act > 0, (CTRL * active).sum(1) / safe, np.nan)
    out["ctrl_active_legacy"] = np.where(n_leg > 0, (CTRL * active_leg).sum(1) / np.maximum(n_leg, 1), np.nan)
    out["active_frac"] = n_act / N
    out["contested_all"] = cont.mean(1)
    out["contested_active"] = np.where(n_act > 0, (cont & active).sum(1) / safe, np.nan)
    for m in vc.MASS_THRS:
        strict_m = cont & (CT >= m) & (T >= m)
        out[f"contested_strict_{vc.mass_tag(m)}"] = np.where(n_act > 0, strict_m.sum(1) / safe, np.nan)
        out[f"contested_strict_all_{vc.mass_tag(m)}"] = strict_m.mean(1)
    out["ct_held_frac_active"] = np.where(n_act > 0, ((CTRL > 0.5) & active).sum(1) / safe, np.nan)
    out["t_held_frac_active"] = np.where(n_act > 0, ((CTRL < 0.5) & active).sum(1) / safe, np.nan)
    for s in ("A", "B"):
        out[f"site_ctrl_{s}"] = CTRL[:, ctx.sites[s]].mean(1) if s in ctx.sites and len(ctx.sites[s]) else np.full(nf, np.nan)
    ca = cont & active
    cs = cont & (CT >= vc.STRICT_MIN_MASS) & (T >= vc.STRICT_MIN_MASS)
    for nm, idx in ctx.name_groups.items():
        out[f"area_ctrl__{nm}"] = CTRL[:, idx].mean(1)
        out[f"area_cont__{nm}"] = ca[:, idx].mean(1)
        out[f"area_contS__{nm}"] = cs[:, idx].mean(1)

    # ------------------------------------------------ positions: occupancy + Voronoi
    pos = vc.read_resolved(vc.T_POSITION / f"{match}.csv", vc.POS_SPEC, "position_maps")
    pk = pos["round"].to_numpy(np.int64) * KEY + pos["frame_idx"].to_numpy(np.int64)
    fid = np.searchsorted(uniq, pk)
    okf = (fid < nf) & (uniq[np.clip(fid, 0, nf - 1)] == pk)
    pos, fid = pos[okf], fid[okf]
    cidx = ctx.idx_of(pos["cur_area"].to_numpy())
    is_ct = (pos["side"].to_numpy() == "CT")
    good = cidx >= 0
    fid, cidx, is_ct = fid[good], cidx[good], is_ct[good]
    gnames = list(ctx.name_groups.keys())
    g_of_node = np.zeros(N, np.int64)
    for gi, nm in enumerate(gnames):
        g_of_node[ctx.name_groups[nm]] = gi
    occ_ct = np.zeros((nf, len(gnames)), np.int16)
    occ_t = np.zeros((nf, len(gnames)), np.int16)
    np.add.at(occ_ct, (fid[is_ct], g_of_node[cidx[is_ct]]), 1)
    np.add.at(occ_t, (fid[~is_ct], g_of_node[cidx[~is_ct]]), 1)
    for gi, nm in enumerate(gnames):
        out[f"occ_ct__{nm}"] = occ_ct[:, gi]
        out[f"occ_t__{nm}"] = occ_t[:, gi]
    vor = np.full(nf, np.nan)
    order = np.argsort(fid, kind="stable")
    fid_s, cidx_s, ct_s = fid[order], cidx[order], is_ct[order]
    bounds = np.searchsorted(fid_s, np.arange(nf + 1))
    inf = np.inf
    for f in range(nf):
        lo, hi = bounds[f], bounds[f + 1]
        if hi == lo:
            continue
        c, m = cidx_s[lo:hi], ct_s[lo:hi]
        dct = ctx.dsym[c[m]].min(0) if m.any() else np.full(N, inf)
        dt = ctx.dsym[c[~m]].min(0) if (~m).any() else np.full(N, inf)
        ct_better, t_better = dct + 1e-9 < dt, dt + 1e-9 < dct
        a_ct, a_t = ctx.tile_area[ct_better].sum(), ctx.tile_area[t_better].sum()
        if a_ct + a_t > 0:
            vor[f] = (a_ct - a_t) / (a_ct + a_t)
    out["vor_margin"] = vor

    fr = pd.DataFrame(out)
    fs = vc.read_resolved(vc.T_FSTATE / f"{match}.csv", {k: [k] for k in FSTATE_KEEP}, "frame_state")
    fr = fr.merge(fs, on=["round", "frame_idx"], how="left")
    fr["bomb_site"] = fr["bomb_site"].fillna("").astype(str)
    fr["bomb_planted"] = fr["bomb_planted"].fillna(False).astype(bool)
    front = [c for c in CORE_COLS if c in fr.columns]
    rest = [c for c in fr.columns if c not in front]
    fr = fr[front + rest]
    for c in fr.columns:
        if fr[c].dtype == np.float64 and c not in ("tick", "t_rel", "tick_rate", "active_thr"):
            fr[c] = fr[c].astype(np.float32)
    return fr


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    vc.add_limit_arg(ap)
    args = ap.parse_args()
    matches = [m for m in vc.match_stems(vc.T_CONTROL, (".csv",))
               if (vc.T_POSITION / f"{m}.csv").exists() and (vc.T_FSTATE / f"{m}.csv").exists()]
    if args.limit:
        matches = matches[: args.limit]
    if not matches:
        raise SystemExit("No matches with control_fields + position_maps + frame_state. "
                         "Run extract_supplementary_tables.py first.")
    ctx = vc.MapContext()
    vc.ensure_dir(vc.T_FSUM)
    done = skipped = failed = 0
    for m in matches:
        out = vc.T_FSUM / f"{m}.parquet"
        if vc.schema_ok(out, CORE_COLS) and vc.stamp_matches(out, "active_thr", vc.ACTIVE_THR) and not args.force:
            skipped += 1
            continue
        try:
            df = build(m, ctx)
            df.to_parquet(out, index=False)
            vc.log(f"done {m}: {len(df)} frames x {df.shape[1]} cols (active_thr={vc.ACTIVE_THR:.4f})")
            done += 1
        except Exception as e:  # noqa: BLE001
            vc.log(f"FAILED {m}: {type(e).__name__}: {e}")
            if out.exists():
                out.unlink()
            failed += 1
    vc.log(f"finished: {done} written, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()
