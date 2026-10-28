#!/usr/bin/env python
"""
Part B, step 1: per-round control signatures over the resolved area names.

For every round, side and window this records the side's *share* of
  - 'field'     : influence mass (ct_mu / t_mu from control_fields), and
  - 'occupancy' : raw player presence (position_maps), the model-free baseline,
summed over nav nodes within each of the resolved areas (assigned_area), averaged over the
frames in the window. Each signature is a distribution over areas (sums to 1).

Windows
  pre_first_loss : [DROP_S, first death) seconds after freeze end; round kept only if the
                   window is at least MIN_WINDOW_S long. 'First death' = first frame where
                   either side's alive count drops below its count in the round's first frame
                   (position_maps holds alive players only, so no kills table is needed).
  early_5_20, mid_20_40 : fixed windows, regardless of deaths.

Time axis: seconds = (tick - freeze_time_end_tick) / TICK_RATE. Frames are 64 ticks apart
(~0.5 s), so frame counts must never be read as seconds.

Inputs : data/de_mirage_area_assignment.csv, data/team_aliases.json,
         outputs/control_fields/*.csv, outputs/position_maps/*.csv, outputs/round_metadata/*.csv
Outputs: outputs/analysis/round_signatures/{match_name}.parquet   (one per match, restartable)
         outputs/analysis/round_signatures.parquet                (all matches combined)

Usage  : python3 build_team_signatures.py --limit 2     # smoke test on two matches
         python3 build_team_signatures.py               # full run
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv

AREA_CSV = "data/de_mirage_area_assignment.csv"
ALIASES_PATH = "data/team_aliases.json"
CF_DIR = "outputs/control_fields"
PM_DIR = "outputs/position_maps"
RM_DIR = "outputs/round_metadata"
OUT_DIR = "outputs/analysis"
PER_MATCH_DIR = f"{OUT_DIR}/round_signatures"
COMBINED_PATH = f"{OUT_DIR}/round_signatures.parquet"

TICK_RATE = 128.0        # demos report 127 or 128; the difference is < 1% of any time here
DROP_S = 5.0             # skip the first 5 s (everyone still in spawn)
MIN_WINDOW_S = 12.0      # pre_first_loss window must be at least this long
FIXED_WINDOWS = {"early_5_20": (5.0, 20.0), "mid_20_40": (20.0, 40.0)}
MIN_FRAMES_FIXED = 6     # at least 3 s of frames inside a fixed window
THRESHOLDS = (20, 25, 30, 40, 60)


# --------------------------------------------------------------------- static inputs
def load_static():
    aa = pd.read_csv(AREA_CSV, usecols=["area_id", "assigned_area"], keep_default_na=False)
    assert aa["area_id"].is_unique, "duplicate area_id in area assignment"
    assert (aa["assigned_area"].str.strip() != "").all(), "blank assigned_area"
    areas = sorted(aa["assigned_area"].unique())
    k_of_area = {a: i for i, a in enumerate(areas)}
    k_of_node = {int(r.area_id): k_of_area[r.assigned_area] for r in aa.itertuples()}

    with open(ALIASES_PATH) as f:
        aliases = json.load(f)
    rm = pd.concat((pd.read_csv(p) for p in glob.glob(f"{RM_DIR}/*.csv")), ignore_index=True)
    assert rm["t_team"].isin(aliases).all() and rm["ct_team"].isin(aliases).all(), \
        "unmapped team names: re-run build_team_aliases.py"
    assert not rm.duplicated(["match_id", "round"]).any(), "duplicate round_metadata keys"
    rm["t_c"] = rm["t_team"].map(aliases)
    rm["ct_c"] = rm["ct_team"].map(aliases)
    return areas, k_of_node, rm


def read_control_field(path):
    opts = pacsv.ConvertOptions(
        include_columns=["round", "frame_idx", "area_id", "ct_mu", "t_mu"],
        column_types={"round": pa.int32(), "frame_idx": pa.int32(), "area_id": pa.int32(),
                      "ct_mu": pa.float32(), "t_mu": pa.float32()},
    )
    return pacsv.read_csv(path, convert_options=opts)


# --------------------------------------------------------------------- one match
def process_match(name, areas, k_of_node, rm_all):
    K, n_nodes = len(areas), len(k_of_node)
    pm = pd.read_csv(f"{PM_DIR}/{name}.csv",
                     usecols=["match_id", "round", "frame_idx", "side", "tick", "current_areaId"])
    assert pm["match_id"].nunique() == 1, "position_maps mixes several matches"
    match_id = pm["match_id"].iloc[0]
    rm = rm_all[rm_all["match_id"] == match_id].set_index("round")
    assert len(rm), f"match_id {match_id} not found in round_metadata"

    # ---- control field -> per-frame mass in each resolved area
    tbl = read_control_field(f"{CF_DIR}/{name}.csv")
    rnd, fr = tbl.column("round").to_numpy(), tbl.column("frame_idx").to_numpy()
    aid = tbl.column("area_id").to_numpy()
    ct, tt = tbl.column("ct_mu").to_numpy(), tbl.column("t_mu").to_numpy()
    assert len(rnd) % n_nodes == 0, "control_fields rows are not a multiple of the node count"
    F = len(rnd) // n_nodes
    aid2, rnd2, fr2 = (x.reshape(F, n_nodes) for x in (aid, rnd, fr))
    assert (aid2 == aid2[0]).all(), "area_id order differs between frames"
    assert (rnd2 == rnd2[:, :1]).all() and (fr2 == fr2[:, :1]).all(), "frame blocks not contiguous"
    M = np.zeros((n_nodes, K), dtype=np.float32)
    for j, a in enumerate(aid2[0]):
        M[j, k_of_node[int(a)]] = 1.0
    FIELD = {"CT": ct.reshape(F, n_nodes) @ M, "T": tt.reshape(F, n_nodes) @ M}

    # ---- frame table: time since freeze end and alive counts (from position_maps)
    ft = pd.DataFrame({"round": rnd2[:, 0], "frame_idx": fr2[:, 0]})
    meta = pm.groupby(["round", "frame_idx"], as_index=False)["tick"].first()
    cnt = pm.groupby(["round", "frame_idx", "side"]).size().unstack("side", fill_value=0).reset_index()
    cnt.columns.name = None
    ft = (ft.merge(meta, on=["round", "frame_idx"], how="left", validate="one_to_one")
            .merge(cnt, on=["round", "frame_idx"], how="left", validate="one_to_one"))
    assert len(ft) == F and ft[["tick", "CT", "T"]].notna().all().all(), \
        "control_fields frames missing from position_maps"
    assert ft.groupby("round")["frame_idx"].apply(lambda s: s.is_monotonic_increasing).all()
    ft["freeze_end"] = ft["round"].map(rm["freeze_time_end_tick"])
    assert ft["freeze_end"].notna().all(), "round missing from round_metadata"
    ft["sec"] = (ft["tick"] - ft["freeze_end"]) / TICK_RATE
    assert ft["sec"].min() > -1.0, "frames before freeze end present"
    for s in ("CT", "T"):
        ft[f"base_{s}"] = ft.groupby("round")[s].transform("first")
    ft["lost"] = (ft["CT"] < ft["base_CT"]) | (ft["T"] < ft["base_T"])
    first_loss = ft[ft["lost"]].groupby("round")["sec"].min()

    # ---- occupancy baseline: players per resolved area, per frame and side
    row_of = pd.Series(np.arange(F), index=pd.MultiIndex.from_arrays([ft["round"], ft["frame_idx"]]))
    occ = (pm.assign(k=pm["current_areaId"].astype(int).map(k_of_node))
             .groupby(["round", "frame_idx", "side", "k"]).size().rename("n").reset_index())
    assert occ["k"].notna().all(), "current_areaId not in area assignment"
    occ["row"] = row_of.reindex(pd.MultiIndex.from_frame(occ[["round", "frame_idx"]])).to_numpy()
    assert not np.isnan(occ["row"]).any()
    OCC = {"CT": np.zeros((F, K)), "T": np.zeros((F, K))}
    for s in ("CT", "T"):
        o = occ[occ["side"] == s]
        np.add.at(OCC[s], (o["row"].astype(int).to_numpy(), o["k"].astype(int).to_numpy()),
                  o["n"].to_numpy())

    # ---- per round, window, side, method
    sec_all = ft["sec"].to_numpy()
    rows = []
    for rnd_no, ix in ft.groupby("round").indices.items():
        sec_r = sec_all[ix]
        dt = float(np.median(np.diff(sec_r))) if len(sec_r) > 1 else 0.5
        t_end = min(float(first_loss.get(rnd_no, np.inf)), float(sec_r.max()) + dt)
        windows = {"pre_first_loss": ((sec_r >= DROP_S) & (sec_r < t_end),
                                      (t_end - DROP_S) >= MIN_WINDOW_S, DROP_S, t_end)}
        for wname, (a, b) in FIXED_WINDOWS.items():
            m = (sec_r >= a) & (sec_r < b)
            windows[wname] = (m, int(m.sum()) >= MIN_FRAMES_FIXED, a, b)
        t_team, ct_team = rm.at[rnd_no, "t_c"], rm.at[rnd_no, "ct_c"]
        for wname, (mask, ok, w0, w1) in windows.items():
            if not ok:
                continue
            fi = ix[mask]
            for side, team, opp in (("CT", ct_team, t_team), ("T", t_team, ct_team)):
                for method, mat in (("field", FIELD[side]), ("occupancy", OCC[side])):
                    sub = mat[fi].astype(np.float64)
                    tot = sub.sum(axis=1)
                    good = tot > 0
                    if not good.any():
                        continue
                    vec = (sub[good] / tot[good, None]).mean(axis=0)
                    rows.append({
                        "match_id": match_id, "match_name": name, "round": int(rnd_no),
                        "side": side, "team": team, "opp": opp, "window": wname,
                        "method": method, "n_frames": int(good.sum()),
                        "win_start": float(w0), "win_end": float(w1),
                        **{f"a_{a}": float(v) for a, v in zip(areas, vec)},
                    })
    df = pd.DataFrame(rows)
    acols = [f"a_{a}" for a in areas]
    assert np.allclose(df[acols].sum(axis=1), 1.0, atol=1e-4), "a signature does not sum to 1"
    return df


# --------------------------------------------------------------------- summary
def js_rows(P, Q):
    """Row-wise Jensen-Shannon distance (base 2) between two arrays of distributions."""
    Mx = 0.5 * (P + Q)

    def kl(a, b):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(a > 0, a * np.log2(a / b), 0.0).sum(axis=1)

    return np.sqrt(np.clip(0.5 * kl(P, Mx) + 0.5 * kl(Q, Mx), 0, None))


def summarise(sig, areas, rm_all):
    acols = [f"a_{a}" for a in areas]
    matches = sig["match_id"].unique()
    n_rounds = int(rm_all["match_id"].isin(matches).sum())
    print(f"\nmatches: {len(matches)} | rounds in round_metadata for those matches: {n_rounds}")
    print("signature rows by window/method:")
    print(sig.groupby(["window", "method"]).size().unstack("method").to_string())

    pre = sig[(sig["window"] == "pre_first_loss") & (sig["method"] == "field")]
    usable = pre.groupby(["match_id", "round"]).ngroups
    print(f"\nrounds with a usable pre_first_loss window (skip {DROP_S:g} s, need >= {MIN_WINDOW_S:g} s): "
          f"{usable} of {n_rounds} ({usable / max(n_rounds, 1):.1%})")
    print("window length (s) over usable rounds:")
    print((pre.drop_duplicates(["match_id", "round"]).eval("win_end - win_start"))
          .describe(percentiles=[.1, .25, .5, .75, .9]).round(1).to_string())
    for side in ("T", "CT"):
        n = pre[pre["side"] == side].groupby("team").size()
        print(f"{side}-side: teams with >= k usable rounds:",
              {k: int((n >= k).sum()) for k in THRESHOLDS}, f"| teams seen: {len(n)}")

    f = sig[(sig["window"] == "pre_first_loss") & (sig["method"] == "field")]
    o = sig[(sig["window"] == "pre_first_loss") & (sig["method"] == "occupancy")]
    key = ["match_id", "round", "side"]
    m = f.merge(o, on=key, suffixes=("_f", "_o"))
    d = js_rows(m[[c + "_f" for c in acols]].to_numpy(), m[[c + "_o" for c in acols]].to_numpy())
    print(f"\nper-round JS distance, field vs occupancy signature (0 = identical, 1 = disjoint): "
          f"median {np.median(d):.3f}, mean {d.mean():.3f}, 90th pct {np.percentile(d, 90):.3f}")
    print("pooled mean share by area (pre_first_loss, field), T side vs CT side:")
    tab = pd.DataFrame({
        "T": f[f["side"] == "T"][acols].mean().to_numpy(),
        "CT": f[f["side"] == "CT"][acols].mean().to_numpy(),
    }, index=areas).sort_values("CT", ascending=False).round(3)
    print(tab.to_string())


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="process only the first N matches")
    args = ap.parse_args()

    os.makedirs(PER_MATCH_DIR, exist_ok=True)
    areas, k_of_node, rm_all = load_static()
    print(f"{len(areas)} resolved areas, {len(k_of_node)} nodes")
    names = sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{CF_DIR}/*.csv"))
    if args.limit:
        names = names[: args.limit]

    n_fail = 0
    for i, name in enumerate(names, 1):
        out = f"{PER_MATCH_DIR}/{name}.parquet"
        if os.path.exists(out):
            print(f"[{i}/{len(names)}] skip (done): {name}")
            continue
        tmp = out + ".tmp"
        try:
            df = process_match(name, areas, k_of_node, rm_all)
            df.to_parquet(tmp, index=False)
            os.replace(tmp, out)
            print(f"[{i}/{len(names)}] done: {name} ({len(df)} rows)")
        except Exception as e:  # keep going; partial files are removed so the match is retried
            n_fail += 1
            print(f"[{i}/{len(names)}] FAILED: {name}: {type(e).__name__}: {e}")
            if os.path.exists(tmp):
                os.remove(tmp)

    files = sorted(glob.glob(f"{PER_MATCH_DIR}/*.parquet"))
    sig = pd.concat((pd.read_parquet(p) for p in files), ignore_index=True)
    sig.to_parquet(COMBINED_PATH, index=False)
    print(f"\nWrote {COMBINED_PATH}: {len(sig)} rows from {len(files)} matches ({n_fail} failures this run)")
    summarise(sig, areas, rm_all)


if __name__ == "__main__":
    main()
