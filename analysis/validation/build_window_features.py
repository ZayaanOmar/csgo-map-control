"""
build_window_features.py   (location-aware predictor, step 1 of 3)
===================================================================
Builds ONE row per round of window-averaged features, from the control_fields CSVs
(columns: round, frame_idx, area_id, ct_mu, t_mu, control_val; one file per match).

WINDOW (per round)
    start  t0 = 5 s after freeze end                      (skips spawn dwell)
    end    t1 = min(first kill, bomb plant, 30 s cap)     (all 10 players alive -> head-count is
                                                           constant, so any signal is positioning)
    keep   rounds with t1 - t0 >= 12 s and >= 8 frames in the window.

FEATURES (all window means over frames with t0 <= t_rel <= t1)
    node level   net[v]  = mean( control_val[v] - 0.5 )    bounded +-0.5, + = CT-held, - = T-held
                 pres[v] = mean( ct_mu[v] + t_mu[v] )       non-directional total influence
    area level   footprint-weighted mean of the node values over the nodes of each area, using the
                 labels in data/de_mirage_area_assignment.csv (weights = |width x height| of each tile)
    aggregate    ca_win = mean(ctrl_active - 0.5)           the existing active-footprint control (M_agg)
                 cf_win = mean(ctrl_full   - 0.5)           used only as a cross-check of the node table
    economy      log_eq_ratio, eq_diff_k, is_pistol         copied from the round table (baseline M0)

NET vs PRESENCE: net says WHO holds a place; presence says HOW MUCH is going on there. net ~ 0 with high
presence = contested; net ~ 0 with low presence = empty. They are reported separately everywhere.

Outputs (outputs/predictor/):
    window_rounds.csv            every round in the round table, with eligibility flags and reasons,
                                 window bounds, economy, ca_win/cf_win, team names (if available),
                                 and keep_legacy (the OLD pre-engagement cap-30 sample, for reproducing M0)
    window_node_net.parquet      rk + one column per nav node (n<area_id>), eligible rounds only
    window_node_presence.parquet same, presence
    window_area_net.csv          rk + one column per area label
    window_area_presence.csv     same, presence
    window_meta.json             parameters used (downstream scripts read this)

Usage (from project root):
    python analysis/validation/build_window_features.py
    python analysis/validation/build_window_features.py --t0 5 --cap 30 --min-len 12
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "pipeline"))          # pipeline/ (for area_assignment)
import validation_common as vc  # noqa: E402
from area_assignment import load_assignment  # noqa: E402

MAP_NAME = "de_mirage"
ECON = ["log_eq_ratio", "eq_diff_k", "is_pistol"]
LEGACY_MIN_PRE, LEGACY_CAP, LEGACY_MIN_FRAMES = 5.0, 30.0, 3     # the old headline window


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def load_footprints() -> dict:
    """|width x height| of every de_mirage nav tile (awpy's own tile area can be negative)."""
    from awpy.data import NAV
    out = {}
    for i, a in NAV[MAP_NAME].items():
        out[int(i)] = abs((a["northWestX"] - a["southEastX"]) * (a["northWestY"] - a["southEastY"]))
    return out


def load_tables():
    rt = vc.load_round_table()
    rt["rk"] = rt["match_name"] + ":" + rt["round"].astype(str)
    need = {"y_ct_win", "first_kill_t", "plant_t", "log_eq_ratio", "eq_diff", "is_pistol_round"}
    missing = need - set(rt.columns)
    if missing:
        raise SystemExit(f"round table lacks {sorted(missing)}")
    rt["eq_diff_k"] = rt["eq_diff"] / 1000.0
    rt["is_pistol"] = rt["is_pistol_round"].astype(float)
    fr = vc.load_frame_summaries(columns=["round", "frame_idx", "t_rel", "ctrl_full", "ctrl_active"])
    fr["rk"] = fr["match_name"] + ":" + fr["round"].astype(str)
    fr = fr[fr["t_rel"] >= 0].copy()
    fr["ca"] = fr["ctrl_active"] - 0.5
    fr["cf"] = fr["ctrl_full"] - 0.5
    return rt, fr


# --------------------------------------------------------------------------- #
# Round selection
# --------------------------------------------------------------------------- #
def select_rounds(rt: pd.DataFrame, fr: pd.DataFrame, args):
    """Add window bounds / counts / eligibility to the round table. Returns (rt, window_frames)."""
    rt = rt.copy()
    fk = rt["first_kill_t"].where(np.isfinite(rt["first_kill_t"]), np.inf).fillna(np.inf)
    pl = rt["plant_t"].where(np.isfinite(rt["plant_t"]), np.inf).fillna(np.inf)
    rt["win_t0"] = args.t0
    rt["win_t1"] = np.minimum(np.minimum(fk, pl), args.cap)
    rt["win_len"] = rt["win_t1"] - rt["win_t0"]

    lim = fr.merge(rt[["rk", "win_t0", "win_t1"]], on="rk", how="inner")
    inwin = lim[(lim["t_rel"] >= lim["win_t0"]) & (lim["t_rel"] <= lim["win_t1"])]
    g = inwin.groupby("rk")
    stats = pd.DataFrame({"n_frames": g.size(), "ca_win": g["ca"].mean(), "cf_win": g["cf"].mean()})
    rt = rt.merge(stats, left_on="rk", right_index=True, how="left")
    rt["n_frames"] = rt["n_frames"].fillna(0).astype(int)

    # legacy sample (old headline window: [0, min(first kill, 30)], first kill >= 5 s, >= 3 frames)
    last_t = fr.groupby("rk")["t_rel"].max()
    rlen = rt["round_len_s"].fillna(rt["rk"].map(last_t)) if "round_len_s" in rt.columns else rt["rk"].map(last_t)
    leg_end = np.minimum(fk, LEGACY_CAP)
    leg = fr.merge(pd.DataFrame({"rk": rt["rk"], "leg_end": leg_end.to_numpy()}), on="rk")
    n_leg = leg[leg["t_rel"] <= leg["leg_end"]].groupby("rk").size()
    rt["keep_legacy"] = ((fk >= LEGACY_MIN_PRE) & (rlen >= leg_end - 1e-9)
                         & (rt["rk"].map(n_leg).fillna(0) >= LEGACY_MIN_FRAMES))

    # eligibility, with the FIRST failing reason recorded
    reasons = np.full(len(rt), "", dtype=object)
    def flag(mask, why):
        m = mask.to_numpy() & (reasons == "")
        reasons[m] = why
    flag(rt["win_len"] < args.min_len, "window_shorter_than_min_len")
    flag(rt["n_frames"] == 0, "no_frames_in_window")
    flag(rt["n_frames"] < args.min_frames, "too_few_frames")
    flag(rt[["y_ct_win"] + ECON].isna().any(axis=1), "missing_label_or_economy")
    rt["drop_reason"] = reasons
    rt["eligible"] = rt["drop_reason"] == ""

    ok = rt.loc[rt["eligible"], ["rk", "match_name", "round", "win_t0", "win_t1"]]
    wf = inwin.merge(ok[["rk"]], on="rk")[["match_name", "round", "frame_idx", "rk"]]
    return rt, wf


# --------------------------------------------------------------------------- #
# Extraction from control_fields
# --------------------------------------------------------------------------- #
def extract_match(path: Path, wf: pd.DataFrame, sep: str, chunksize: int) -> pd.DataFrame:
    """Window-mean net / presence per (round, area_id) for one match."""
    usecols = ["round", "frame_idx", "area_id", "ct_mu", "t_mu", "control_val"]
    wf = wf[["round", "frame_idx"]].astype({"round": "int64", "frame_idx": "int64"})
    parts = []
    for ch in pd.read_csv(path, sep=sep, usecols=usecols, chunksize=chunksize,
                          dtype={"round": "int64", "frame_idx": "int64", "area_id": "int64",
                                 "ct_mu": "float64", "t_mu": "float64", "control_val": "float64"}):
        m = ch.merge(wf, on=["round", "frame_idx"], how="inner")
        if m.empty:
            continue
        m["net"] = m["control_val"] - 0.5
        m["pres"] = m["ct_mu"] + m["t_mu"]
        parts.append(m.groupby(["round", "area_id"], sort=False)
                      .agg(net=("net", "sum"), pres=("pres", "sum"), n=("net", "size")))
    if not parts:
        return pd.DataFrame()
    tot = pd.concat(parts).groupby(level=[0, 1]).sum()
    tot["net"] /= tot["n"]
    tot["pres"] /= tot["n"]
    return tot


def build_node_tables(rt_ok: pd.DataFrame, wf: pd.DataFrame, node_ids: list, ctrl_dir: Path,
                      sep: str, chunksize: int):
    """Return (net_df, pres_df) wide (rk x nodes) + list of matches whose file was missing/empty."""
    nets, press, bad = [], [], []
    matches = sorted(rt_ok["match_name"].unique())
    for k, mn in enumerate(matches, 1):
        path = ctrl_dir / f"{mn}.csv"
        if not path.exists():
            bad.append((mn, "file missing"))
            continue
        tot = extract_match(path, wf[wf["match_name"] == mn], sep, chunksize)
        if tot.empty:
            bad.append((mn, "no window frames found in control_fields (frame_idx mismatch?)"))
            continue
        for col, store in (("net", nets), ("pres", press)):
            w = tot[col].unstack("area_id").reindex(columns=node_ids)
            w.index = [f"{mn}:{r}" for r in w.index]
            store.append(w.astype("float32"))
        if k % 20 == 0 or k == len(matches):
            vc.log(f"  extracted {k}/{len(matches)} matches")
    net = pd.concat(nets) if nets else pd.DataFrame(columns=node_ids)
    pres = pd.concat(press) if press else pd.DataFrame(columns=node_ids)
    return net, pres, bad


# --------------------------------------------------------------------------- #
# Area aggregation
# --------------------------------------------------------------------------- #
def area_matrix(node_ids: list, labels: dict, foot: dict):
    """W_raw (nodes x areas) of raw footprint weights, and the area names."""
    areas = sorted(set(labels[i] for i in node_ids))
    idx = {a: j for j, a in enumerate(areas)}
    W = np.zeros((len(node_ids), len(areas)))
    for r, i in enumerate(node_ids):
        W[r, idx[labels[i]]] = foot[i]
    return W, areas


def aggregate_to_areas(node_df: pd.DataFrame, W: np.ndarray, areas: list) -> pd.DataFrame:
    """Footprint-weighted, NaN-aware mean of node values per area."""
    M = node_df.to_numpy(float)
    ok = np.isfinite(M)
    num = np.where(ok, M, 0.0) @ W
    den = ok.astype(float) @ W
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0, num / den, np.nan)
    return pd.DataFrame(out, index=node_df.index, columns=areas)


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
def verify(rt: pd.DataFrame, net: pd.DataFrame, pres: pd.DataFrame, W: np.ndarray, areas: list,
           area_net: pd.DataFrame, area_pres: pd.DataFrame, foot: dict, node_ids: list) -> None:
    vc.log("\nVERIFICATION")
    ok = rt[rt["rk"].isin(net.index)].set_index("rk").loc[net.index]

    cov = net.notna().mean(axis=1)
    vc.log(f"  node coverage per round: min {cov.min():.3f}, mean {cov.mean():.3f} "
           f"({'OK' if cov.min() == 1.0 else 'WARNING: some nodes missing from control_fields'})")

    colsum = (W / W.sum(axis=0, keepdims=True)).sum(axis=0)
    assert np.allclose(colsum, 1.0), "area weights do not sum to 1"
    vc.log(f"  area weights: {len(areas)} areas, every column sums to 1  -> OK")

    # hand re-aggregation of one random round
    rng = np.random.default_rng(0)
    rk = net.index[rng.integers(len(net))]
    for df_node, df_area, tag in ((net, area_net, "net"), (pres, area_pres, "presence")):
        v = df_node.loc[rk].to_numpy(float)
        j = int(rng.integers(len(areas)))
        m = W[:, j] > 0
        hand = float(np.nansum(v[m] * W[m, j]) / W[m, j][np.isfinite(v[m])].sum())
        assert np.isclose(hand, df_area.loc[rk].iloc[j], atol=1e-6), (tag, hand, df_area.loc[rk].iloc[j])
    vc.log(f"  hand re-aggregation of a random round/area matches the table (net and presence) -> OK")

    # cross-check against the existing frame summaries (also catches frame_idx misalignment)
    # ctrl_full in frame_summaries is an UNWEIGHTED mean over nodes, so the unweighted node mean must match it
    # exactly. The footprint-weighted mean is printed for context only: it differs by design.
    cf = ok["cf_win"].to_numpy(float)
    node_mean = net.to_numpy(float).mean(axis=1)
    c = np.corrcoef(node_mean, cf)[0, 1]
    d = np.abs(node_mean - cf).max()
    flag = "OK - node table lines up with frame_summaries" if (c > 0.999 and d < 1e-3) else \
        "WARNING: mismatch - check frame_idx alignment between control_fields and frame_summaries"
    vc.log(f"  cf_win (frame_summaries) vs UNWEIGHTED node mean: corr {c:.4f}, max |diff| {d:.4f}  -> {flag}")
    w_all = np.array([foot[i] for i in node_ids], float)
    node_wmean = (np.nan_to_num(net.to_numpy(float)) * w_all).sum(axis=1) / w_all.sum()
    cw = np.corrcoef(node_wmean, cf)[0, 1]
    vc.log(f"  (context) vs footprint-weighted node mean: corr {cw:.4f} - lower by design, because large tiles "
           f"dominate the weighted mean while ctrl_full counts every tile equally")


def report_selection(rt: pd.DataFrame, args) -> None:
    vc.log(f"\nROUND SELECTION: {len(rt)} rounds in the round table")
    for why, n in rt["drop_reason"].replace("", "eligible").value_counts().items():
        vc.log(f"  {why:<32} {n:>6}")
    e = rt[rt["eligible"]]
    vc.log(f"  eligible: {len(e)} rounds from {e['match_name'].nunique()} matches; "
           f"CT win rate {e['y_ct_win'].mean():.3f}")
    vc.log(f"  window length: median {e['win_len'].median():.1f}s, "
           f"{(e['win_t1'] >= args.cap).mean():.1%} of windows reach the {args.cap:g}s cap, "
           f"median frames {e['n_frames'].median():.0f}")
    n_leg = int(rt["keep_legacy"].sum())
    both = int((rt["keep_legacy"] & rt["eligible"]).sum())
    vc.log(f"  legacy sample (old cap-30 headline, first kill >= 5 s): {n_leg} rounds; "
           f"{both} of them are also eligible here, {int(rt['eligible'].sum()) - both} eligible rounds are not in it")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--t0", type=float, default=5.0, help="seconds after freeze end to start the window")
    ap.add_argument("--cap", type=float, default=30.0, help="window end cap (s); use a large value for uncapped")
    ap.add_argument("--min-len", type=float, default=12.0, help="minimum window length (s)")
    ap.add_argument("--min-frames", type=int, default=8)
    ap.add_argument("--control-dir", default="outputs/control_fields")
    ap.add_argument("--assignment", default="data/de_mirage_area_assignment.csv")
    ap.add_argument("--out-dir", default="outputs/predictor")
    ap.add_argument("--sep", default=",", help="field separator of the control_fields CSVs")
    ap.add_argument("--chunksize", type=int, default=2_000_000)
    args = ap.parse_args()

    out = Path(args.out_dir)
    vc.ensure_dir(out)
    labels, _ = load_assignment(args.assignment)
    node_ids = sorted(labels)
    foot = load_footprints()
    assert set(node_ids) <= set(foot), "assignment contains ids not in NAV"

    rt, fr = load_tables()
    vc.log(f"{len(rt)} rounds, {rt['match_name'].nunique()} matches, {len(fr)} frames")
    rt, wf = select_rounds(rt, fr, args)
    report_selection(rt, args)

    vc.log("\nextracting node-level window means from control_fields ...")
    elig = rt[rt["eligible"]]
    net, pres, bad = build_node_tables(elig, wf, node_ids, Path(args.control_dir), args.sep, args.chunksize)
    for mn, why in bad:
        vc.log(f"  WARNING: dropping match {mn}: {why}")
    have = set(net.index)
    rt.loc[rt["eligible"] & ~rt["rk"].isin(have), "drop_reason"] = "no_control_fields_data"
    rt["eligible"] = rt["rk"].isin(have)
    net, pres = net.loc[sorted(have)], pres.loc[sorted(have)]
    vc.log(f"  {len(net)} eligible rounds with node data")

    W, areas = area_matrix(node_ids, labels, foot)
    area_net = aggregate_to_areas(net, W, areas)
    area_pres = aggregate_to_areas(pres, W, areas)
    verify(rt, net, pres, W, areas, area_net, area_pres, foot, node_ids)

    # ----- write
    keep_cols = ["rk", "match_name", "round", "y_ct_win", "eligible", "drop_reason", "keep_legacy",
                 "win_t0", "win_t1", "win_len", "n_frames", "ca_win", "cf_win",
                 "first_kill_t", "plant_t"] + ECON
    for c in ("ct_team", "t_team"):
        if c in rt.columns:
            keep_cols.append(c)
    if "ct_team" not in rt.columns:
        vc.log("  NOTE: round table has no ct_team/t_team columns; team-pair folds and team rating "
               "will need them (add from round_metadata).")
    rt[keep_cols].to_csv(out / "window_rounds.csv", index=False)
    for name, df in (("window_node_net", net), ("window_node_presence", pres)):
        d = df.copy()
        d.columns = [f"n{c}" for c in d.columns]
        d.insert(0, "rk", d.index)
        d.reset_index(drop=True).to_parquet(out / f"{name}.parquet", index=False)
    area_net.rename_axis("rk").reset_index().to_csv(out / "window_area_net.csv", index=False)
    area_pres.rename_axis("rk").reset_index().to_csv(out / "window_area_presence.csv", index=False)
    (out / "window_meta.json").write_text(json.dumps(
        dict(t0=args.t0, cap=args.cap, min_len=args.min_len, min_frames=args.min_frames,
             sigma=500.0, net="mean(control_val-0.5)", presence="mean(ct_mu+t_mu)",
             n_rounds=int(len(net)), n_nodes=len(node_ids), areas=areas, assignment=args.assignment), indent=2))
    vc.log(f"\nWrote {out}/ window_rounds.csv, window_node_net.parquet, window_node_presence.parquet, "
           f"window_area_net.csv, window_area_presence.csv, window_meta.json")


if __name__ == "__main__":
    main()
