"""
validation_common.py
====================
Shared helpers for the validity / extension analyses (kill-duel validity,
plant-aligned control, early-window prediction, win-probability time series,
side/site favourability, location importance).

Design rules (inherited from the existing pipeline)
---------------------------------------------------
* Flat-by-type outputs:  outputs/<table_name>/<match_name>.<ext>
* Every per-match builder is skip-if-exists AND schema-aware (a cached file
  that lacks a required column is recomputed, never silently trusted) and
  deletes partial output on failure.
* No awpy import is needed anywhere in this suite: area names, tile
  rectangles and centroids are read from the patched-graph pickle's node
  attributes, and distances from the APSP pickle.
* Scripts are run from the PROJECT ROOT (override with env CSGO_ROOT).

Everything that depends on an assumption about an upstream file's columns goes
through `resolve()` / `read_resolved()` so a mismatch fails loudly with the
list of columns that *are* present.
"""
from __future__ import annotations

import json
import math
import os
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from scipy.special import expit

import matplotlib

matplotlib.use("Agg")

# --------------------------------------------------------------------- paths
ROOT = Path(os.environ.get("CSGO_ROOT", ".")).resolve()
DATA = ROOT / "data"
OUT = ROOT / "outputs"
FILTERED_DEMOS = DATA / "filtered_demos"
APSP_PATH = DATA / "de_mirage_apsp_dict.pkl"
GRAPH_PATH = DATA / "de_mirage_patched_graph_v3.pkl"
BOMBSITE_PATH = DATA / "de_mirage_bombsite_nodes.json"

# existing pipeline tables
T_POSITION = OUT / "position_maps"
T_CONTROL = OUT / "control_fields"
T_ROUND_META = OUT / "round_metadata"
T_PLANTS = OUT / "bomb_plants"
# tables created by this suite
T_ECON = OUT / "round_economy"
T_KILLS = OUT / "kill_events"
T_FSTATE = OUT / "frame_state"
T_FSUM = OUT / "frame_summaries"
T_ENG_ADV = OUT / "engagement_adv"
T_ENG_LOC = OUT / "engagement_loc"
T_ENG_SER = OUT / "engagement_series"
METRICS = OUT / "metrics"

# ----------------------------------------------------------- model constants
MAP_NAME = "de_mirage"
SIGMA = float(os.environ.get("SIGMA", 500.0))          # primary sigma (project)
K_SIGMOID = 1.0                                         # sigmoid steepness (project)
# Active footprint: a node is "active" when ct_mu + t_mu > ACTIVE_THR (a PRE-sigmoid threshold).
# Default exp(-2) = 0.135 = the influence of a single player standing exactly 2 sigma away.
# The project's original value was exp(-4.5) (3 sigma) -> LEGACY_ACTIVE_THR, kept only for the
# Metric 3 reproduction check. Changing ACTIVE_THR makes cached frame_summaries/engagement tables stale:
# they carry an `active_thr` stamp column and are rebuilt automatically when it differs.
ACTIVE_THR = float(os.environ.get("ACTIVE_THR", math.exp(-2.0)))
LEGACY_ACTIVE_THR = math.exp(-4.5)
EPSILON = ACTIVE_THR                                    # backwards-compatible alias
# "Robustly contested": BOTH teams must carry at least this much influence mass at the node
# (1.0 = one full-strength player). Sensitivity values are precomputed for MASS_THRS.
STRICT_MIN_MASS = 0.5
MASS_THRS = (0.25, 0.5, 1.0)
DEFAULT_BOOT = int(os.environ.get("N_BOOT", 1000))      # default bootstrap resamples
CONTEST_LO, CONTEST_HI = 0.4, 0.6                       # Metric 1 contested band
INF_CLIP = 1e9                                          # inf -> 1e9 before mu
# MUST equal Z_TOL in position_to_node_map.py (see --check-mapper in the extractor)
Z_TOL = float(os.environ.get("Z_TOL", 250.0))
KDTREE_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))

# --------------------------------------------------------------- column specs
# canonical name -> acceptable actual names (first hit wins)
POS_SPEC = {
    "round": ["round"], "frame_idx": ["frame_idx"], "player_name": ["player_name"],
    "side": ["side"], "cur_area": ["current_areaId", "curr_areaId"],
    "proj_area": ["projected_areaId", "proj_areaId"],
}
POS_XYZ_SPEC = {"x": ["current_x", "x"], "y": ["current_y", "y"], "z": ["current_z", "z"]}
CF_SPEC = {
    "round": ["round"], "frame_idx": ["frame_idx"],
    "area_id": ["area_id", "areaId", "area"],
    "ct_mu": ["ct_mu", "CT_mu", "ct", "CT"], "t_mu": ["t_mu", "T_mu", "t", "T"],
    "control_val": ["control_val", "control_value", "control"],
}
RM_SPEC = {
    "round": ["round"], "winning_side": ["winning_side"],
    "ct_team": ["ct_team"], "t_team": ["t_team"],
    "end_tick": ["end_tick"], "freeze_time_end_tick": ["freeze_time_end_tick"],
    "round_end_reason": ["round_end_reason"],
}
BP_SPEC = {"round": ["round"], "bomb_site": ["bomb_site"], "tick": ["tick"]}


# ------------------------------------------------------------------- general
def log(*a):
    print(*a, flush=True)


def ensure_dir(p: Path) -> Path:
    Path(p).mkdir(parents=True, exist_ok=True)
    return Path(p)


def safe_name(s) -> str:
    s = "" if s is None or (isinstance(s, float) and math.isnan(s)) else str(s)
    return "UNNAMED" if s.strip() == "" else re.sub(r"[^A-Za-z0-9]", "_", s)


def match_stems(directory: Path, suffixes=(".csv", ".parquet", ".json.xz")) -> list[str]:
    directory = Path(directory)
    if not directory.exists():
        return []
    out = []
    for p in sorted(directory.iterdir()):
        for sfx in suffixes:
            if p.name.endswith(sfx):
                out.append(p.name[: -len(sfx)])
                break
    return sorted(set(out))


def table_path(table_dir: Path, match: str, ext: str = ".csv") -> Path:
    return Path(table_dir) / f"{match}{ext}"


def mass_tag(t: float) -> str:
    return f"{t:.2f}".replace(".", "p")


def stamp_matches(path: Path, col: str, value: float, tol: float = 1e-9) -> bool:
    """True iff the cached parquet's constant stamp column equals `value` (cache-invalidation guard)."""
    try:
        import pyarrow.parquet as pq
        v = pq.ParquetFile(path).read(columns=[col]).column(0)[0].as_py()
        return abs(float(v) - float(value)) <= tol
    except Exception:
        return False


def columns_of(path: Path) -> list[str]:
    path = Path(path)
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        return list(pq.ParquetFile(path).schema_arrow.names)
    return list(pd.read_csv(path, nrows=0).columns)


def schema_ok(path: Path, required: list[str]) -> bool:
    """True iff `path` exists and contains every required column."""
    path = Path(path)
    if not path.exists():
        return False
    try:
        have = set(columns_of(path))
    except Exception:
        return False
    return all(c in have for c in required)


def resolve(df: pd.DataFrame, spec: dict, where="table", optional=()):
    rename = {}
    for canon, cands in spec.items():
        hit = next((c for c in cands if c in df.columns), None)
        if hit is None:
            if canon in optional:
                continue
            raise KeyError(f"{where}: none of {cands} found for '{canon}'. "
                           f"Columns present: {list(df.columns)}")
        rename[hit] = canon
    return df.rename(columns=rename)


def read_resolved(path: Path, spec: dict, where="table", optional=(), extra_cols=()):
    """Read only the columns named in `spec` (+extra_cols) and rename to canonical names."""
    path = Path(path)
    have = columns_of(path)
    use, rename = [], {}
    for canon, cands in spec.items():
        hit = next((c for c in cands if c in have), None)
        if hit is None:
            if canon in optional:
                continue
            raise KeyError(f"{where} ({path.name}): none of {cands} found for "
                           f"'{canon}'. Columns present: {have}")
        use.append(hit)
        rename[hit] = canon
    use += [c for c in extra_cols if c in have and c not in use]
    if path.suffix == ".parquet":
        df = pd.read_parquet(path, columns=use)
    else:
        df = pd.read_csv(path, usecols=use)
    return df.rename(columns=rename)


def concat_tables(table_dir: Path, matches=None, columns=None, ext=None) -> pd.DataFrame:
    """Concatenate per-match tables, adding a `match_name` column."""
    table_dir = Path(table_dir)
    parts = []
    for m in (matches if matches is not None else match_stems(table_dir)):
        p = None
        for e in ([ext] if ext else [".parquet", ".csv"]):
            if (table_dir / f"{m}{e}").exists():
                p = table_dir / f"{m}{e}"
                break
        if p is None:
            continue
        df = pd.read_parquet(p, columns=columns) if p.suffix == ".parquet" else pd.read_csv(p, usecols=columns)
        df.insert(0, "match_name", m)
        parts.append(df)
    if not parts:
        raise FileNotFoundError(f"No tables found in {table_dir}")
    return pd.concat(parts, ignore_index=True)


def add_limit_arg(ap):
    ap.add_argument("--limit", type=int, default=None, help="process only the first N matches")
    ap.add_argument("--force", action="store_true", help="recompute even if outputs exist")
    return ap


# --------------------------------------------------------------- map context
class MapContext:
    """APSP matrix + nav-graph attributes in a single consistent index order
    (the APSP file's index_to_area_map order, i.e. sorted area IDs)."""

    def __init__(self):
        with open(APSP_PATH, "rb") as f:
            apsp = pickle.load(f)
        self.dist = np.asarray(apsp["distance_matrix"], dtype=np.float64)
        ita = {int(k): int(v) for k, v in apsp["index_to_area_map"].items()}
        n = len(ita)
        self.n = n
        self.area_ids = np.array([ita[i] for i in range(n)], dtype=np.int64)
        self.ati = {int(a): i for i, a in enumerate(self.area_ids)}
        self._lookup = np.full(int(self.area_ids.max()) + 1, -1, dtype=np.int64)
        self._lookup[self.area_ids] = np.arange(n)
        with open(GRAPH_PATH, "rb") as f:
            self.G = pickle.load(f)
        raw_names, centers, tile_area = [], [], []
        for a in self.area_ids:
            nd = self.G.nodes[int(a)]
            raw_names.append(nd.get("areaName", ""))
            centers.append(nd["center"])
            tile_area.append(abs((nd["northWestX"] - nd["southEastX"]) *
                                 (nd["northWestY"] - nd["southEastY"])))
        self.raw_names = np.array(raw_names, dtype=object)
        self.names = np.array([safe_name(s) for s in raw_names], dtype=object)
        self.centers = np.asarray(centers, dtype=np.float64)
        self.tile_area = np.asarray(tile_area, dtype=np.float64)
        self.dsym = np.minimum(self.dist, self.dist.T)
        self.name_groups = {nm: np.where(self.names == nm)[0] for nm in sorted(set(self.names))}
        with open(BOMBSITE_PATH, "r") as f:
            bs = json.load(f)
        self.sites = {k: np.array([self.ati[int(a)] for a in v if int(a) in self.ati], dtype=np.int64)
                      for k, v in bs.items() if k in ("A", "B")}

    def idx_of(self, area_ids) -> np.ndarray:
        """Vectorised area-ID -> matrix index (-1 if unknown)."""
        a = np.asarray(area_ids)
        out = np.full(a.shape, -1, dtype=np.int64)
        ok = np.isfinite(a.astype(float)) & (a >= 0) & (a < len(self._lookup))
        out[ok] = self._lookup[a[ok].astype(np.int64)]
        return out

    def mu_matrix(self, cur_idx, proj_idx, sigma: float = SIGMA) -> np.ndarray:
        """Per-player influence over all nodes — identical to inf_comp():
        d_eff = (D[cur,:] + D[proj,:]) / 2 ; inf -> 1e9 ; mu = exp(-d_eff^2 / 2 sigma^2)."""
        d = (self.dist[cur_idx] + self.dist[proj_idx]) / 2.0
        d = np.where(np.isinf(d), INF_CLIP, d)
        return np.exp(-(d ** 2) / (2.0 * sigma ** 2))


class NodeMapper:
    """Two-tier KDTree mapping used by the project: nearest of k=5 centroids
    whose centroid-z is within z_tol of the query z; else raw nearest."""

    def __init__(self, ctx: MapContext, z_tol: float = Z_TOL, k: int = 5, workers: int = KDTREE_WORKERS):
        self.ctx, self.z_tol, self.k, self.workers = ctx, z_tol, min(k, ctx.n), workers
        self.tree = KDTree(ctx.centers)

    def map(self, coords: np.ndarray):
        coords = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
        idx = np.full(len(coords), -1, dtype=np.int64)
        dist = np.full(len(coords), np.nan)
        ok = np.isfinite(coords).all(axis=1)
        if ok.any():
            q = coords[ok]
            d, ix = self.tree.query(q, k=self.k, workers=self.workers)
            d, ix = d.reshape(len(q), -1), ix.reshape(len(q), -1)
            zok = np.abs(self.ctx.centers[ix, 2] - q[:, None, 2]) <= self.z_tol
            pick = np.where(zok.any(axis=1), zok.argmax(axis=1), 0)
            r = np.arange(len(q))
            idx[ok] = ix[r, pick]
            dist[ok] = d[r, pick]
        return idx, dist


# --------------------------------------------------------------------- stats
def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (p, c - h, c + h)


def bh_fdr(p) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (NaN-safe)."""
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    pv = p[ok]
    m = len(pv)
    if m == 0:
        return out
    order = np.argsort(pv)
    ranked = pv[order] * m / (np.arange(m) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(m)
    adj[order] = np.minimum(ranked, 1.0)
    out[ok] = adj
    return out


def cohens_d(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    sp = math.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else float("nan")


def logit_newton(X, y, ridge: float = 1e-6, max_iter: int = 60, tol: float = 1e-8):
    """Fast logistic regression (Newton/IRLS, tiny ridge) with intercept.
    Returns (coef_without_intercept, intercept). ~2-3x faster than sklearn per fit (no estimator
    overhead), which matters in 1000+ resample bootstraps; agrees with LogisticRegression(C=1e6) to ~3e-4."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, p = X.shape
    A = np.column_stack([X, np.ones(n)])
    beta = np.zeros(p + 1)
    eye = ridge * np.eye(p + 1)
    for _ in range(max_iter):
        mu = expit(A @ beta)
        w = np.clip(mu * (1 - mu), 1e-9, None)
        g = A.T @ (y - mu) - ridge * beta
        step = np.linalg.solve((A * w[:, None]).T @ A + eye, g)
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    return beta[:-1], float(beta[-1])


def group_rows(groups) -> list[np.ndarray]:
    """Row-index arrays, one per distinct group (for cluster bootstrap)."""
    codes, uniq = pd.factorize(np.asarray(groups))
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(len(uniq) + 1))
    return [order[bounds[i]:bounds[i + 1]] for i in range(len(uniq))]


def cluster_bootstrap(groups, stat_fn, n_boot=500, seed=0) -> np.ndarray:
    """Resample whole groups (matches) with replacement; stat_fn(row_idx) -> scalar/array."""
    gi = group_rows(groups)
    rng = np.random.default_rng(seed)
    G = len(gi)
    out = []
    for _ in range(n_boot):
        rows = np.concatenate([gi[i] for i in rng.integers(0, G, G)])
        out.append(stat_fn(rows))
    return np.asarray(out, dtype=float)


def pctl_ci(arr, alpha=0.05):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lo, hi = np.nanpercentile(arr, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0)
    return lo, hi


def boot_pvalue(boot, null=0.0) -> float:
    """Two-sided bootstrap p-value for H0: parameter == null."""
    b = np.asarray(boot, float)
    b = b[np.isfinite(b)]
    if len(b) == 0:
        return float("nan")
    p = 2 * min((b <= null).mean(), (b >= null).mean())
    return float(min(max(p, 1.0 / (len(b) + 1)), 1.0))


# ------------------------------------------------------------ results registry
REGISTRY = METRICS / "results_registry.json"


def _clean(v):
    if v is None:
        return None
    v = float(v)
    return None if (math.isnan(v) or math.isinf(v)) else v


def record_result(key: str, estimate, lo=None, hi=None, n=None, note: str = ""):
    """Append/overwrite one named headline result (estimate, CI, n) in the results registry
    (outputs/metrics/results_registry.json), so every reported quantity is in one machine-readable file."""
    ensure_dir(REGISTRY.parent)
    reg = json.loads(REGISTRY.read_text()) if REGISTRY.exists() else {}
    reg[key] = {"estimate": _clean(estimate), "lo": _clean(lo), "hi": _clean(hi),
                "n": None if n is None else int(n), "note": note}
    REGISTRY.write_text(json.dumps(reg, indent=1, sort_keys=True))


# ------------------------------------------------------- round-level assembly
def tick_rates() -> dict:
    out = {}
    for m in match_stems(T_FSTATE, (".csv",)):
        d = pd.read_csv(T_FSTATE / f"{m}.csv", usecols=["tick_rate"], nrows=1)
        out[m] = float(d["tick_rate"].iloc[0])
    return out


def load_round_table(matches=None) -> pd.DataFrame:
    """One row per (match_name, round): outcome, economy, first-kill time, plant info."""
    rm_parts = []
    for m in (matches if matches is not None else match_stems(T_ROUND_META, (".csv",))):
        p = T_ROUND_META / f"{m}.csv"
        if not p.exists():
            continue
        d = read_resolved(p, RM_SPEC, "round_metadata",
                          optional=("ct_team", "t_team", "end_tick", "freeze_time_end_tick", "round_end_reason"))
        d.insert(0, "match_name", m)
        rm_parts.append(d)
    rt = pd.concat(rm_parts, ignore_index=True)
    rt = rt[rt["winning_side"].isin(["CT", "T"])].copy()
    rt["y_ct_win"] = (rt["winning_side"] == "CT").astype(int)

    if T_ECON.exists():
        econ = concat_tables(T_ECON, matches=sorted(rt["match_name"].unique()), ext=".csv")
        econ = econ.drop(columns=[c for c in ("match_id",) if c in econ.columns])
        rt = rt.merge(econ, on=["match_name", "round"], how="left", suffixes=("", "_econ"))
    rates = tick_rates()
    rt["tick_rate"] = rt["match_name"].map(rates)

    if T_KILLS.exists():
        kills = concat_tables(T_KILLS, matches=sorted(rt["match_name"].unique()), ext=".csv",
                              columns=["round", "t_rel"])
        fk = kills.groupby(["match_name", "round"], as_index=False)["t_rel"].min().rename(columns={"t_rel": "first_kill_t"})
        rt = rt.merge(fk, on=["match_name", "round"], how="left")
    else:
        rt["first_kill_t"] = np.nan

    if T_PLANTS.exists():
        pl = []
        for m in sorted(rt["match_name"].unique()):
            p = T_PLANTS / f"{m}.csv"
            if p.exists():
                d = read_resolved(p, BP_SPEC, "bomb_plants")
                d.insert(0, "match_name", m)
                pl.append(d)
        if pl:
            pl = pd.concat(pl, ignore_index=True).sort_values("tick")
            pl["plant_site"] = pl["bomb_site"].astype(str).str.upper().str[-1]
            pl = pl.groupby(["match_name", "round"], as_index=False).first()[["match_name", "round", "tick", "plant_site"]]
            pl = pl.rename(columns={"tick": "plant_tick"})
            rt = rt.merge(pl, on=["match_name", "round"], how="left")
    for c in ("plant_tick", "plant_site"):
        if c not in rt.columns:
            rt[c] = np.nan
    if "freeze_time_end_tick" in rt.columns:
        rt["plant_t"] = (rt["plant_tick"] - rt["freeze_time_end_tick"]) / rt["tick_rate"]
    else:
        rt["plant_t"] = np.nan
    return rt


def load_frame_summaries(columns=None, matches=None) -> pd.DataFrame:
    return concat_tables(T_FSUM, matches=matches, columns=columns, ext=".parquet")


def savefig(fig, path: Path, dpi=150):
    ensure_dir(Path(path).parent)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    import matplotlib.pyplot as plt
    plt.close(fig)
    log(f"  wrote {path}")


def boot_binned_mean(groups, bins, values, n_bins, n_boot=500, seed=0):
    """Mean of `values` within integer bins 0..n_bins-1 with a cluster (match) bootstrap CI.
    Uses per-group sufficient statistics, so it is fast even for millions of rows.
    Returns (mean, lo, hi, count) arrays of length n_bins."""
    values = np.asarray(values, float)
    bins = np.asarray(bins)
    codes, uniq = pd.factorize(np.asarray(groups))
    G = len(uniq)
    ok = np.isfinite(values) & (bins >= 0) & (bins < n_bins)
    S = np.zeros((G, n_bins))
    C = np.zeros((G, n_bins))
    np.add.at(S, (codes[ok], bins[ok].astype(int)), values[ok])
    np.add.at(C, (codes[ok], bins[ok].astype(int)), 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(C.sum(0) > 0, S.sum(0) / C.sum(0), np.nan)
    rng = np.random.default_rng(seed)
    boots = np.empty((n_boot, n_bins))
    for b in range(n_boot):
        pick = rng.integers(0, G, G)
        cs = C[pick].sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            boots[b] = np.where(cs > 0, S[pick].sum(0) / cs, np.nan)
    lo, hi = pctl_ci(boots)
    return mean, lo, hi, C.sum(0)
