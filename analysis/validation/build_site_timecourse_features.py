"""
build_site_timecourse_features.py   (site-choice timing analysis, step 1 of 2)
==============================================================================
How EARLY is the site choice readable from positioning? This builds area-level window features for several window
ENDS in a single pass over the control_fields CSVs, for a FIXED set of rounds, so a later script can plot
prediction quality against window end without the sample changing under it.

FIXED SAMPLE
    exactly the rounds used by analysis_location_site.py (eligible rounds from --feat-dir with a plant at A or B
    that comes >= --min-lead seconds after the 30 s window ends). Smaller window ends only make the lead larger.

WINDOWS (cumulative)
    for each cap c:   frames with  t0 <= t_rel <= min(c, first kill, plant)      (t0 = 5 s, as in the main analysis)
    so cap 30 is IDENTICAL to the main site analysis (checked below against window_area_net.csv), and a smaller
    cap simply stops earlier. Rounds whose first kill or plant comes before a cap use their shorter window.

HOW (one pass)
    every window frame is tagged with the smallest cap that includes it; per (round, cap tag, area) the
    footprint-weighted sums of (control_val - 0.5), (ct_mu + t_mu) and of the weights are accumulated, then
    cumulated over caps, so window mean = cumulative weighted sum / cumulative weight (exactly the
    footprint-weighted area mean of build_window_features.py, with the same net / presence definitions).

Outputs (outputs/predictor_timecourse/):
    area_net_cap<c>.csv, area_pres_cap<c>.csv   rk + one column per area, for every cap
    rounds.csv                                  rk, match_name, round, site, y_site_B, win_end_cap<c>, n_frames_cap<c>
    timecourse_meta.json

Usage (project root):
    python analysis/validation/build_site_timecourse_features.py
    python analysis/validation/build_site_timecourse_features.py --caps 8,12,16,20,24,28
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
sys.path.insert(0, str(HERE.parents[1] / "pipeline"))
import validation_common as vc  # noqa: E402
import build_window_features as bwf  # noqa: E402
import analysis_location_site as als  # noqa: E402
from area_assignment import load_assignment  # noqa: E402


def fmt(c: float) -> str:
    return f"{c:g}"


def reference_sample(feat_dir: Path, bomb_dir: Path, min_lead: float) -> pd.DataFrame:
    allr = pd.read_csv(feat_dir / "window_rounds.csv")
    el = allr[allr["eligible"]].copy()
    el = als.make_filter(bomb_dir, min_lead)(el)
    return el.reset_index(drop=True)


def frames_for_sample(sample: pd.DataFrame, rt: pd.DataFrame, fr: pd.DataFrame, caps: np.ndarray, t0: float):
    """Window frames (t0 <= t_rel <= min(first kill, plant, max cap)), each tagged with its smallest including cap."""
    info = rt[rt["rk"].isin(set(sample["rk"]))][["rk", "first_kill_t", "plant_t"]].copy()
    fk = info["first_kill_t"].where(np.isfinite(info["first_kill_t"]), np.inf).fillna(np.inf)
    pl = info["plant_t"].where(np.isfinite(info["plant_t"]), np.inf).fillna(np.inf)
    info["t1max"] = np.minimum(fk, pl)
    f = fr[fr["rk"].isin(set(sample["rk"]))].merge(info[["rk", "t1max"]], on="rk")
    f = f[(f["t_rel"] >= t0) & (f["t_rel"] <= np.minimum(f["t1max"], caps.max()))].copy()
    f["cap_idx"] = np.searchsorted(caps, f["t_rel"].to_numpy(), side="left")
    return f, info.set_index("rk")["t1max"]


def stream_match(path: Path, wf: pd.DataFrame, area_of: np.ndarray, weight_of: np.ndarray, n_caps: int,
                 n_areas: int, sep: str, chunksize: int):
    """Return dict round -> (sn, sp, sw) arrays of shape (n_caps, n_areas)."""
    usecols = ["round", "frame_idx", "area_id", "ct_mu", "t_mu", "control_val"]
    wf = wf[["round", "frame_idx", "cap_idx"]].astype("int64")
    parts = []
    for ch in pd.read_csv(path, sep=sep, usecols=usecols, chunksize=chunksize,
                          dtype={"round": "int64", "frame_idx": "int64", "area_id": "int64",
                                 "ct_mu": "float64", "t_mu": "float64", "control_val": "float64"}):
        m = ch.merge(wf, on=["round", "frame_idx"], how="inner")
        if m.empty:
            continue
        ids = m["area_id"].to_numpy()
        m["ai"] = area_of[ids]
        w = weight_of[ids]
        m["w"] = w
        m["wnet"] = w * (m["control_val"].to_numpy() - 0.5)
        m["wpres"] = w * (m["ct_mu"].to_numpy() + m["t_mu"].to_numpy())
        parts.append(m.groupby(["round", "cap_idx", "ai"], sort=False)
                      .agg(sn=("wnet", "sum"), sp=("wpres", "sum"), sw=("w", "sum")))
    if not parts:
        return {}
    tot = pd.concat(parts).groupby(level=[0, 1, 2]).sum().reset_index()
    out = {}
    for r, g in tot.groupby("round"):
        sn, sp, sw = (np.zeros((n_caps, n_areas)) for _ in range(3))
        sn[g["cap_idx"], g["ai"]] = g["sn"]
        sp[g["cap_idx"], g["ai"]] = g["sp"]
        sw[g["cap_idx"], g["ai"]] = g["sw"]
        out[int(r)] = (sn, sp, sw)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feat-dir", default="outputs/predictor", help="features used by the main site analysis")
    ap.add_argument("--bomb-dir", default="outputs/bomb_plants")
    ap.add_argument("--min-lead", type=float, default=10.0)
    ap.add_argument("--control-dir", default="outputs/control_fields")
    ap.add_argument("--assignment", default="data/de_mirage_area_assignment.csv")
    ap.add_argument("--caps", default="8,10,12,15,18,21,24,27,30", help="comma-separated window ends (s)")
    ap.add_argument("--t0", type=float, default=5.0)
    ap.add_argument("--out-dir", default="outputs/predictor_timecourse")
    ap.add_argument("--sep", default=",")
    ap.add_argument("--chunksize", type=int, default=2_000_000)
    args = ap.parse_args()

    caps = np.array(sorted(float(c) for c in args.caps.split(",")))
    out = Path(args.out_dir)
    vc.ensure_dir(out)
    feat = Path(args.feat_dir)
    ref_meta = json.loads((feat / "window_meta.json").read_text())

    labels, _ = load_assignment(args.assignment)
    node_ids = sorted(labels)
    foot = bwf.load_footprints()
    areas = sorted(set(labels.values()))
    aidx = {a: j for j, a in enumerate(areas)}
    size = max(node_ids) + 1
    area_of, weight_of = np.full(size, -1, dtype=np.int64), np.zeros(size)
    for i in node_ids:
        area_of[i] = aidx[labels[i]]
        weight_of[i] = foot[i]

    sample = reference_sample(feat, Path(args.bomb_dir), args.min_lead)
    vc.log(f"fixed sample: {len(sample)} rounds from {sample['match_name'].nunique()} matches; "
           f"caps {', '.join(fmt(c) for c in caps)} (t0 = {args.t0:g}s)")
    rt, fr = bwf.load_tables()
    wf_all, t1max = frames_for_sample(sample, rt, fr, caps, args.t0)
    vc.log(f"{len(wf_all)} window frames (frame_summaries) will be extracted")

    n_caps, n_areas = len(caps), len(areas)
    # sums: per round -> (cumulated over caps later)
    rk_sums = {}
    matches = sorted(sample["match_name"].unique())
    for k, mn in enumerate(matches, 1):
        path = Path(args.control_dir) / f"{mn}.csv"
        if not path.exists():
            raise SystemExit(f"control_fields file missing for match {mn}: {path}")
        res = stream_match(path, wf_all[wf_all["match_name"] == mn], area_of, weight_of, n_caps, n_areas,
                           args.sep, args.chunksize)
        for r, v in res.items():
            rk_sums[f"{mn}:{r}"] = v
        if k % 20 == 0 or k == len(matches):
            vc.log(f"  extracted {k}/{len(matches)} matches")

    missing = [rk for rk in sample["rk"] if rk not in rk_sums]
    if missing:
        raise SystemExit(f"{len(missing)} sample rounds have no control_fields rows (frame_idx mismatch?), "
                         f"e.g. {missing[:3]}")

    # cumulate over caps -> window means for every cap
    rks = list(sample["rk"])
    NET = np.full((n_caps, len(rks), n_areas), np.nan)
    PRES = np.full_like(NET, np.nan)
    for i, rk in enumerate(rks):
        sn, sp, sw = (np.cumsum(a, axis=0) for a in rk_sums[rk])
        with np.errstate(invalid="ignore", divide="ignore"):
            NET[:, i, :] = np.where(sw > 0, sn / sw, np.nan)
            PRES[:, i, :] = np.where(sw > 0, sp / sw, np.nan)
    vc.log(f"non-finite feature values: net {int((~np.isfinite(NET)).sum())}, presence {int((~np.isfinite(PRES)).sum())}"
           f"  (should be 0)")

    # rounds table: window end and frame counts per cap
    nfr = (wf_all.groupby(["rk", "cap_idx"]).size().unstack(fill_value=0)
           .reindex(index=rks, columns=range(n_caps), fill_value=0).cumsum(axis=1))
    rounds = sample[["rk", "match_name", "round", "site", "y_site_B"]].copy()
    t1 = rounds["rk"].map(t1max).to_numpy()
    for j, c in enumerate(caps):
        rounds[f"win_end_cap{fmt(c)}"] = np.minimum(t1, c)
        rounds[f"n_frames_cap{fmt(c)}"] = nfr[j].to_numpy()
        pd.DataFrame(NET[j], columns=areas).assign(rk=rks)[["rk"] + areas].to_csv(out / f"area_net_cap{fmt(c)}.csv", index=False)
        pd.DataFrame(PRES[j], columns=areas).assign(rk=rks)[["rk"] + areas].to_csv(out / f"area_pres_cap{fmt(c)}.csv", index=False)
    rounds.to_csv(out / "rounds.csv", index=False)
    (out / "timecourse_meta.json").write_text(json.dumps(dict(
        caps=[float(c) for c in caps], t0=args.t0, areas=areas, n_rounds=len(rks), reference_feat_dir=str(feat),
        min_lead=args.min_lead), indent=2))

    # ---- verification against the main analysis' features (cap 30 must be identical)
    vc.log("\nVERIFICATION")
    ref_cap = float(ref_meta["cap"])
    if ref_cap in set(caps) and abs(float(ref_meta["t0"]) - args.t0) < 1e-9:
        j = int(np.flatnonzero(caps == ref_cap)[0])
        for name, arr, fname in (("net", NET[j], "window_area_net.csv"), ("presence", PRES[j], "window_area_presence.csv")):
            fpath = feat / fname
            if not fpath.exists():
                continue
            ref = pd.read_csv(fpath).set_index("rk").loc[rks, areas].to_numpy(float)
            d = np.abs(arr - ref).max()
            vc.log(f"  cap {fmt(ref_cap)} {name}: max |new - window_area_{name}.csv| = {d:.2e}  "
                   f"-> {'OK' if d < 1e-4 else 'MISMATCH: investigate before using the timecourse'}")
    else:
        vc.log(f"  reference features use cap {ref_cap:g} / t0 {ref_meta['t0']}: no identical cap to compare against")
    vc.log("  mean window frames per cap: " + ", ".join(
        f"{fmt(c)}s: {rounds[f'n_frames_cap{fmt(c)}'].mean():.0f}" for c in caps))
    vc.log(f"\nWrote {out}/")


if __name__ == "__main__":
    main()
