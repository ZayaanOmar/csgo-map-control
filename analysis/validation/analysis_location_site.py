"""
analysis_location_site.py   (location-aware predictor: target = WHICH SITE THE T SIDE PLANTS AT)
=============================================================================================
Does early positioning encode strategy? y = 1 if the first bomb plant of the round is at B, 0 if at A
(planted rounds only), predicted from the same early-window features as the round-winner analysis.
The engine lives in location_predictor_lib.py; this script supplies the rows, the label and the primary tests.

ROWS
    eligible rounds (window = 5 s after freeze end to min(first kill, plant, cap); see build_window_features.py)
    that have a plant at A or B AND whose plant comes at least --min-lead seconds (default 10) after the window ENDS.
    The lead guard stops the window from containing the plant itself or T standing on the site.
    Plant sites come from outputs/bomb_plants/<match_name>.csv (columns: round, bomb_site, tick, ...);
    the first plant (lowest tick) per round counts; 'A'/'B' are read from the END of the bomb_site string.
    The kept sample is a slow-execute subset: fast rushes and early picks are excluded by construction.

MODELS (same grid as the winner analysis; M0 = economy, M_agg = aggregate control, M1/M2 = area/node net control,
        offset models M0+M1/M0+M2; --presence adds the net+presence rows)
PRIMARY tests (Holm-adjusted together): M1 vs M_const and M2 vs M_const, where M_const is the A/B base rate
    (does location-resolved control predict the site at all?). Descriptive: M0 vs M_const (does the buy state?),
    M2 vs M1, M0+M1 / M0+M2 vs M0, presence rows.

READING IT
    By ~30 s the T side has often committed, so this measures how readable the execute is from positioning, not
    pre-decision intent. An AUC far above ~0.9 deserves a leakage check. area_coefficients.csv reports odds ratios
    for B (vs A) per +0.1 net: net is NEGATIVE where T holds an area, so a negative coefficient on net means
    'T control there points to B'. A weak result would also be informative: early positioning would not carry the
    site decision at this resolution.

Outputs (outputs/metrics/location_site/ by default): same files as the winner analysis.

Usage (project root):
    python analysis/validation/analysis_location_site.py --presence
    python analysis/validation/analysis_location_site.py --grouping teampair
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validation_common as vc  # noqa: E402
import location_predictor_lib as lib  # noqa: E402

PRIMARY_SITE = [("M1", "M_const"), ("M2", "M_const")]


def load_sites(bomb_dir: Path) -> pd.DataFrame:
    """First plant per (match_name, round) with its site, from <bomb_dir>/<match_name>.csv."""
    files = sorted(Path(bomb_dir).glob("*.csv"))
    if not files:
        raise SystemExit(f"no bomb_plants CSVs found in {bomb_dir}")
    parts = []
    for f in files:
        d = pd.read_csv(f, usecols=["round", "bomb_site", "tick"])
        d["match_name"] = f.stem
        parts.append(d)
    b = pd.concat(parts, ignore_index=True).sort_values("tick")
    raw = b["bomb_site"].astype(str).str.strip()
    vc.log("  bomb_site raw values: " + ", ".join(f"{k!r}: {v}" for k, v in raw.value_counts().items()))
    up = raw.str.upper()
    b["site"] = np.where(up.str.endswith("A"), "A", np.where(up.str.endswith("B"), "B", ""))
    if (b["site"] == "").any():
        vc.log(f"  WARNING: {(b['site'] == '').sum()} plant rows have an unrecognised bomb_site and are ignored")
    b = b[b["site"] != ""]
    return b.groupby(["match_name", "round"], as_index=False).first()[["match_name", "round", "site"]]


def make_filter(bomb_dir: Path, min_lead: float):
    def f(el: pd.DataFrame) -> pd.DataFrame:
        el = el.merge(load_sites(bomb_dir), on=["match_name", "round"], how="left")
        n_all = len(el)
        planted = el["site"].isin(["A", "B"])
        lead_ok = el["plant_t"].notna() & ((el["plant_t"] - el["win_t1"]) >= min_lead)
        keep = planted & lead_ok
        vc.log(f"SITE TARGET: {n_all} eligible rounds; {int(planted.sum())} with a plant; "
               f"{int((planted & ~lead_ok).sum())} dropped because the plant is < {min_lead:g}s after the window ends; "
               f"{int(keep.sum())} kept")
        el = el[keep].copy()
        vc.log(f"  kept rounds: A {int((el['site'] == 'A').sum())}, B {int((el['site'] == 'B').sum())}")
        el["y_site_B"] = (el["site"] == "B").astype(int)
        return el
    return f


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feat-dir", default="outputs/predictor")
    ap.add_argument("--folds", default=None, help="default: <feat-dir>/folds.csv")
    ap.add_argument("--grouping", choices=["match", "teampair"], default="match")
    ap.add_argument("--repeats", type=int, default=None, help="default: all repeats in folds.csv")
    ap.add_argument("--n-boot", type=int, default=vc.DEFAULT_BOOT)
    ap.add_argument("--presence", action="store_true", help="add the net+presence rows")
    ap.add_argument("--bomb-dir", default="outputs/bomb_plants")
    ap.add_argument("--min-lead", type=float, default=10.0,
                    help="plant must come >= this many seconds after the window ends")
    ap.add_argument("--out-dir", default="outputs/metrics/location_site")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    feat = Path(args.feat_dir)
    out = Path(args.out_dir + ("" if args.grouping == "match" else "_teampair"))
    meta = json.loads((feat / "window_meta.json").read_text())
    vc.log(f"window: t0={meta['t0']}s cap={meta['cap']}s min_len={meta['min_len']}s; "
           f"{meta['n_rounds']} eligible rounds, {meta['n_nodes']} nodes, {len(meta['areas'])} areas")

    D = lib.load_data(feat, args.presence, round_filter=make_filter(Path(args.bomb_dir), args.min_lead),
                      label_col="y_site_B")
    folds = pd.read_csv(args.folds or feat / "folds.csv")
    repeats = args.repeats or int(folds["repeat"].nunique())
    vc.log(f"{D.n} rounds, {len(D.mnames)} matches; grouping={args.grouping}; {repeats} repeats x "
           f"{folds['fold_match'].nunique()} folds; share of plants at B {D.y.mean():.3f}")

    lib.run_grid(D, folds, args.grouping, repeats, out, PRIMARY_SITE, presence=args.presence, n_boot=args.n_boot,
                 seed=args.seed, y_col="y_site_B", record_prefix="locpred_site",
                 meta=dict(target="site", feat_dir=str(feat), min_lead=args.min_lead, bomb_dir=args.bomb_dir,
                           window=meta))


if __name__ == "__main__":
    main()
