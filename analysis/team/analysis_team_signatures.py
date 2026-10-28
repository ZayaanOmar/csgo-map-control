#!/usr/bin/env python
"""
Part B, step 2: are control signatures team-specific, and does the influence field add
anything over raw player occupancy?

Input : outputs/analysis/round_signatures.parquet   (from build_team_signatures.py)

Unit of resampling = one team's rounds in one match on one side ("unit"). Splits are by
match, never by round: rounds within a match share an opponent and a game plan, so a
round-level split would inflate how identifiable a team looks.

Analyses (every window x side x method; PRIMARY_WINDOW is the confirmatory one)
  1. Team identification. Split each team's matches into two random halves. For each
     team, is its half-A signature closer (Jensen-Shannon distance) to its own half-B
     than to every other team's half-B?  Compared with a null in which team labels are
     permuted across units (unit sizes kept).  Also within-team vs between-team distance.
  2. (primary window) team x area heatmap of deviation from the pooled signature,
     ordered by hierarchical clustering of the JS distance matrix.
  3. (primary window) bootstrap stability of each team's nearest neighbour.
  4. (primary window) field vs occupancy: Mantel-style Spearman correlation between the
     two team-distance matrices.

Outputs (outputs/analysis/):
  signature_identification.csv, team_signatures/*.csv, team_neighbours_*.csv,
  figures/signature_heatmap_{side}_{method}.png
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

SIG_PATH = "outputs/analysis/round_signatures.parquet"
OUT_DIR = "outputs/analysis"
PRIMARY_WINDOW = "pre_first_loss"
WINDOWS = ("pre_first_loss", "early_5_20", "mid_20_40")
SIDES = ("T", "CT")
METHODS = ("field", "occupancy")

MIN_ROUNDS = 30        # usable rounds per (team, side, window) to be included
MIN_HALF_ROUNDS = 8    # rounds each half needs for a team to count in a split
R_STAT = 20            # random match-level splits averaged into ONE test statistic
N_PERM = 1000          # label permutations; each recomputes the same statistic (R_STAT splits)
N_BOOT = 200           # bootstrap resamples for nearest-neighbour stability
N_MANTEL = 5000
SEED = 42


# ------------------------------------------------------------------ helpers
def js_matrix(P, Q):
    """Jensen-Shannon distance (base 2, in [0, 1]) between every row of P and every row of Q."""
    Pm, Qm = P[:, None, :], Q[None, :, :]
    Mx = 0.5 * (Pm + Qm)
    with np.errstate(divide="ignore", invalid="ignore"):
        kp = np.where(Pm > 0, Pm * np.log2(Pm / Mx), 0.0).sum(-1)
        kq = np.where(Qm > 0, Qm * np.log2(Qm / Mx), 0.0).sum(-1)
    return np.sqrt(np.clip(0.5 * (kp + kq), 0, None))


def units(sig, acols, window, side, method):
    """One row per (team, match): summed per-round signatures and round count; eligible teams only."""
    d = sig[(sig["window"] == window) & (sig["side"] == side) & (sig["method"] == method)]
    g = d.groupby(["team", "match_id"])
    u = g[acols].sum().reset_index()
    u["n"] = g.size().to_numpy()
    tot = u.groupby("team")["n"].sum()
    u = u[u["team"].isin(tot[tot >= MIN_ROUNDS].index)].reset_index(drop=True)
    tid, teams = pd.factorize(u["team"])
    return u, tid, np.asarray(teams), u[acols].to_numpy(float), u["n"].to_numpy(float)


def team_sigs(tid, S, n, n_teams):
    T = np.zeros((n_teams, S.shape[1]))
    np.add.at(T, tid, S)
    N = np.bincount(tid, weights=n, minlength=n_teams)
    return T / N[:, None], N


# ------------------------------------------------------------------ 1. identification
def split_signatures(tid, S, n, n_teams, rng):
    """Random match-level half split per team. Returns (A, B, ok): half signatures of the
    teams that have at least MIN_HALF_ROUNDS rounds in both halves."""
    order = np.lexsort((rng.random(len(tid)), tid))
    t_sorted = tid[order]
    first = np.searchsorted(t_sorted, np.arange(n_teams))
    half = (np.arange(len(tid)) - first[t_sorted]) % 2
    key = t_sorted * 2 + half
    H = np.zeros((2 * n_teams, S.shape[1]))
    np.add.at(H, key, S[order])
    N = np.bincount(key, weights=n[order], minlength=2 * n_teams)
    H, N = H.reshape(n_teams, 2, -1), N.reshape(n_teams, 2)
    ok = (N >= MIN_HALF_ROUNDS).all(axis=1)
    return H[ok, 0] / N[ok, 0, None], H[ok, 1] / N[ok, 1, None], ok


def identification(tid, S, n, n_teams, rng):
    A, B, _ = split_signatures(tid, S, n, n_teams, rng)
    m = len(A)
    if m < 5:
        return None
    D = js_matrix(A, B)
    own = np.diag(D)
    rank = 1 + (D < own[:, None]).sum(axis=1)
    within = float(own.mean())
    between = float((D.sum() - own.sum()) / (m * (m - 1)))
    return {"m": m, "chance": 1.0 / m,
            "acc": float(np.mean(D.argmin(axis=1) == np.arange(m))),
            "mrr": float(np.mean(1.0 / rank)),
            "within": within, "between": between, "gap": between - within}


def stat_over_splits(tid, S, n, n_teams, rng):
    """The test statistic: the mean of each identification measure over R_STAT random splits.
    Observed data and every label permutation use this identical definition."""
    res = [r for r in (identification(tid, S, n, n_teams, rng) for _ in range(R_STAT)) if r]
    return pd.DataFrame(res).mean().to_dict() if res else None


def run_identification(tid, S, n, n_teams, rng):
    obs = stat_over_splits(tid, S, n, n_teams, rng)
    if obs is None:
        return None
    null = [stat_over_splits(rng.permutation(tid), S, n, n_teams, rng) for _ in range(N_PERM)]
    z = pd.DataFrame([x for x in null if x])
    if z.empty:
        return None
    tol = 1e-12
    return {
        "teams_in_split": float(obs["m"]), "chance": float(obs["chance"]),
        "acc": float(obs["acc"]), "acc_null_mean": float(z["acc"].mean()),
        "acc_null_p95": float(z["acc"].quantile(0.95)),
        "p_acc": float((1 + (z["acc"] >= obs["acc"] - tol).sum()) / (1 + len(z))),
        "mrr": float(obs["mrr"]), "mrr_null_mean": float(z["mrr"].mean()),
        "within": float(obs["within"]), "between": float(obs["between"]),
        "gap": float(obs["gap"]), "gap_null_p95": float(z["gap"].quantile(0.95)),
        "p_gap": float((1 + (z["gap"] >= obs["gap"] - tol).sum()) / (1 + len(z))),
    }


def holm(p):
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


# ------------------------------------------------------------------ 2-4. primary-window outputs
def heatmap(TS, N, teams, areas, path, title):
    pooled = (TS * N[:, None]).sum(0) / N.sum()
    D = js_matrix(TS, TS)
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(D, checks=False), "average")
    n_t = len(teams)
    fig = plt.figure(figsize=(11, 0.3 * n_t + 3.2))
    ax_d = fig.add_axes([0.02, 0.19, 0.09, 0.73])
    dn = dendrogram(Z, orientation="left", ax=ax_d, no_labels=True, color_threshold=0,
                    above_threshold_color="0.4")
    ax_d.axis("off")
    order = dn["leaves"][::-1]                      # heatmap rows run top to bottom
    col_order = np.argsort(-pooled)
    dev = (TS[order] - pooled)[:, col_order]
    vmax = float(np.abs(dev).max())
    ax = fig.add_axes([0.27, 0.19, 0.58, 0.73])
    im = ax.imshow(dev, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_yticks(range(n_t))
    ax.set_yticklabels([f"{teams[i]} (n={int(N[i])})" for i in order], fontsize=8)
    ax.set_xticks(range(len(areas)))
    ax.set_xticklabels([areas[j] for j in col_order], rotation=90, fontsize=8)
    cax = fig.add_axes([0.87, 0.19, 0.015, 0.73])
    fig.colorbar(im, cax=cax, label="team share - pooled share")
    fig.suptitle(title, fontsize=10)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def nn_stability(tid, S, n, n_teams, rng):
    TS, _ = team_sigs(tid, S, n, n_teams)
    D = js_matrix(TS, TS)
    np.fill_diagonal(D, np.inf)
    nn_full = D.argmin(axis=1)
    by_team = [np.flatnonzero(tid == t) for t in range(n_teams)]
    hits = np.zeros(n_teams)
    for _ in range(N_BOOT):
        idx = np.concatenate([rng.choice(ix, len(ix)) for ix in by_team])
        TSb, _ = team_sigs(tid[idx], S[idx], n[idx], n_teams)
        Db = js_matrix(TSb, TSb)
        np.fill_diagonal(Db, np.inf)
        hits += Db.argmin(axis=1) == nn_full
    two = np.argsort(D, axis=1)[:, :2]
    return pd.DataFrame({
        "nearest": [None] * n_teams, "nn_distance": D[np.arange(n_teams), two[:, 0]],
        "second": [None] * n_teams, "second_distance": D[np.arange(n_teams), two[:, 1]],
        "nn_stability": hits / N_BOOT,
    }), two


def mantel(Da, Db, rng):
    iu = np.triu_indices_from(Da, k=1)
    obs = spearmanr(Da[iu], Db[iu])[0]
    null = np.empty(N_MANTEL)
    for i in range(N_MANTEL):
        p = rng.permutation(len(Db))
        null[i] = spearmanr(Da[iu], Db[p][:, p][iu])[0]
    return float(obs), float((1 + (null >= obs).sum()) / (1 + N_MANTEL))


# ------------------------------------------------------------------ main
def main():
    rng = np.random.default_rng(SEED)
    for sub in ("team_signatures", "figures"):
        os.makedirs(f"{OUT_DIR}/{sub}", exist_ok=True)
    sig = pd.read_parquet(SIG_PATH)
    acols = [c for c in sig.columns if c.startswith("a_")]
    areas = [c[2:] for c in acols]

    rows, primary = [], {}
    for window in WINDOWS:
        for side in SIDES:
            for method in METHODS:
                u, tid, teams, S, n = units(sig, acols, window, side, method)
                if len(teams) < 5:
                    print(f"skip {window}/{side}/{method}: only {len(teams)} eligible teams")
                    continue
                TS, N = team_sigs(tid, S, n, len(teams))
                tab = pd.DataFrame(TS, columns=acols)
                tab.insert(0, "n_rounds", N.astype(int))
                tab.insert(0, "team", teams)
                tab.to_csv(f"{OUT_DIR}/team_signatures/team_signatures_{side}_{window}_{method}.csv", index=False)
                res = run_identification(tid, S, n, len(teams), rng)
                if res is None:
                    print(f"skip identification {window}/{side}/{method}: too few teams per split")
                    continue
                rows.append({"window": window, "side": side, "method": method,
                             "n_teams": len(teams), "n_units": len(u),
                             "primary": window == PRIMARY_WINDOW, **res})
                if window == PRIMARY_WINDOW:
                    primary[(side, method)] = (teams, tid, S, n, TS, N)
    ident = pd.DataFrame(rows)
    for pcol in ("p_acc", "p_gap"):
        ident[f"{pcol}_holm"] = np.nan
        prim = ident["primary"]
        if prim.any():
            ident.loc[prim, f"{pcol}_holm"] = holm(ident.loc[prim, pcol])
    ident.to_csv(f"{OUT_DIR}/signature_identification.csv", index=False)

    cols = ["window", "side", "method", "n_teams", "teams_in_split", "chance", "acc", "acc_null_p95",
            "p_acc", "p_acc_holm", "mrr", "within", "between", "gap", "gap_null_p95", "p_gap", "p_gap_holm"]
    show = ident[cols].copy()
    num = [c for c in cols[3:] if not c.startswith("p_")]
    show[num] = show[num].round(3)
    for pc in [c for c in cols if c.startswith("p_")]:
        show[pc] = show[pc].round(4)
    print(f"\n=== Team identification: statistic = mean over {R_STAT} match-level splits; "
          f"null = {N_PERM} label permutations (smallest possible p = {1 / (1 + N_PERM):.4f}) ===")
    print(show.to_string(index=False))
    print("\nacc = share of teams whose half-A signature is nearest their own half-B "
          "(chance = 1 / teams_in_split).\ngap = mean between-team minus within-team JS distance.\n"
          "*_holm = Holm-adjusted over the 4 confirmatory (primary-window) cells only; other windows are exploratory.")

    for (side, method), (teams, tid, S, n, TS, N) in primary.items():
        heatmap(TS, N, teams, areas, f"{OUT_DIR}/figures/signature_heatmap_{side}_{method}.png",
                f"{side} side, {method}, {PRIMARY_WINDOW}: team signature minus pooled signature")
        nn, two = nn_stability(tid, S, n, len(teams), rng)
        nn["nearest"], nn["second"] = teams[two[:, 0]], teams[two[:, 1]]
        nn.insert(0, "team", teams)
        nn.to_csv(f"{OUT_DIR}/team_neighbours_{side}_{method}.csv", index=False)
        print(f"\n[{side}/{method}] {len(teams)} teams; mean nearest-neighbour bootstrap stability "
              f"{nn['nn_stability'].mean():.2f} (share of resamples with the same nearest neighbour)")

    print(f"\n=== Field vs occupancy ({PRIMARY_WINDOW}): Spearman correlation of team-distance matrices ===")
    for side in SIDES:
        if (side, "field") in primary and (side, "occupancy") in primary:
            tf, *_, TSf, _ = primary[(side, "field")]
            to, *_, TSo, _ = primary[(side, "occupancy")]
            if list(tf) != list(to):
                print(f"{side}: team sets differ between methods; skipped")
                continue
            Df, Do = js_matrix(TSf, TSf), js_matrix(TSo, TSo)
            rho, p = mantel(Df, Do, rng)
            print(f"{side}: rho = {rho:.3f} (Mantel p = {p:.4f}; {len(tf)} teams; "
                  f"mean team-to-team distance field {Df[np.triu_indices_from(Df, 1)].mean():.3f} "
                  f"vs occupancy {Do[np.triu_indices_from(Do, 1)].mean():.3f})")
    print(f"\nWrote {OUT_DIR}/signature_identification.csv, team_signatures/, team_neighbours_*.csv, figures/")


if __name__ == "__main__":
    main()
