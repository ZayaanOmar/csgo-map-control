"""
Team site preference (T side).

Q1  Do teams differ in P(A | plant) beyond chance and match-to-match variation?
Q2  Shrunk per-team estimates of P(A | plant) with 90% credible intervals.
Q3  Is a team's preference stable (split-half) or noise?

Inputs : outputs/round_metadata/*.csv, outputs/bomb_plants/*.csv, data/team_aliases.json
Outputs: outputs/analysis/team_site_preference.csv
         outputs/analysis/team_site_results.json
         outputs/analysis/figures/team_site_preference_forest.png
"""
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import betaln, expit, logit
from scipy.stats import beta as beta_dist

ALIASES_PATH = "data/team_aliases.json"
OUT_DIR = "outputs/analysis"
MIN_PLANTS = 20        # per team, to enter the test and the prior fit
N_PERM = 10_000
N_SPLITS = 200
N_NULL_SPLITS = 2_000
MIN_HALF_PLANTS = 5    # per half, for a team to enter the split-half correlation
SEED = 42


# ---------------------------------------------------------------- loading
def load_rounds() -> pd.DataFrame:
    rm = pd.concat(
        (pd.read_csv(p) for p in glob.glob("outputs/round_metadata/*.csv")),
        ignore_index=True,
    )
    bp = pd.concat(
        (pd.read_csv(p) for p in glob.glob("outputs/bomb_plants/*.csv")),
        ignore_index=True,
    )
    with open(ALIASES_PATH) as f:
        aliases = json.load(f)

    assert rm["t_team"].notna().all(), "null t_team"
    missing = sorted(set(rm["t_team"]) - set(aliases))
    assert not missing, f"unmapped team names (re-run build_team_aliases.py): {missing}"
    key = ["match_id", "round"]
    assert not rm.duplicated(key).any(), "duplicate round_metadata keys"
    assert not bp.duplicated(key).any(), "duplicate bomb_plants keys"

    rm["team"] = rm["t_team"].map(aliases)
    df = rm.merge(bp[key + ["bomb_site"]], on=key, how="left", validate="one_to_one")
    assert df["bomb_site"].notna().sum() == len(bp), "bomb_plants rows missing from round_metadata"
    assert set(df["bomb_site"].dropna()) <= {"A", "B"}, "unexpected bomb_site values"

    df["planted"] = df["bomb_site"].notna()
    df["is_A"] = (df["bomb_site"] == "A").astype(int)
    df["is_pistol"] = df["round"].isin([1, 16])
    return df


def prep(df: pd.DataFrame):
    """Team-match units (one row per team per match, T-side plants only), eligible teams only."""
    p = df[df["planted"]]
    u = p.groupby(["team", "match_id"]).agg(n=("is_A", "size"), a=("is_A", "sum")).reset_index()
    tot = u.groupby("team")["n"].sum()
    u = u[u["team"].isin(tot[tot >= MIN_PLANTS].index)].reset_index(drop=True)
    tid, teams = pd.factorize(u["team"])
    return u, tid, np.asarray(teams), u["a"].to_numpy(float), u["n"].to_numpy(float)


# ---------------------------------------------------------------- Q1: heterogeneity
def chi2_stat(tid, a, n, p0, n_teams):
    A = np.bincount(tid, weights=a, minlength=n_teams)
    N = np.bincount(tid, weights=n, minlength=n_teams)
    ok = N > 0
    return np.sum((A[ok] - N[ok] * p0) ** 2 / (N[ok] * p0 * (1 - p0)))


def heterogeneity_test(tid, a, n, n_teams, rng):
    """Permute team labels across team-match units (keeps match-level clustering in the null)."""
    p0 = a.sum() / n.sum()
    obs = chi2_stat(tid, a, n, p0, n_teams)
    null = np.array([chi2_stat(rng.permutation(tid), a, n, p0, n_teams) for _ in range(N_PERM)])
    return {
        "p0": float(p0),
        "stat": float(obs),
        "null_mean": float(null.mean()),
        "null_p95": float(np.percentile(null, 95)),
        "p_value": float((1 + np.sum(null >= obs)) / (1 + N_PERM)),
    }


def dispersion_phi(tid, a, n, n_teams):
    """Quasi-binomial dispersion of team-match units around their own team mean.
    phi > 1 means rounds cluster within matches (effective sample size is smaller)."""
    A = np.bincount(tid, weights=a, minlength=n_teams)
    N = np.bincount(tid, weights=n, minlength=n_teams)
    p_hat = (A / N)[tid]
    var = n * p_hat * (1 - p_hat)
    ok = var > 0
    chi2 = np.sum((a[ok] - n[ok] * p_hat[ok]) ** 2 / var[ok])
    k = np.bincount(tid[ok], minlength=n_teams)
    df_ = np.sum(np.maximum(k - 1, 0))
    return float(max(1.0, chi2 / df_)) if df_ > 0 else 1.0


# ---------------------------------------------------------------- Q2: empirical Bayes
def fit_beta_prior(A, N):
    """MLE of a Beta(alpha, beta) prior on team P(A) via the beta-binomial marginal
    likelihood. A, N may be non-integer (effective counts)."""

    def nll(theta):
        m, kappa = expit(theta[0]), np.exp(theta[1])
        al, be = m * kappa, (1 - m) * kappa
        return -np.sum(betaln(A + al, N - A + be) - betaln(al, be))

    res = minimize(
        nll,
        x0=[logit(A.sum() / N.sum()), np.log(20.0)],
        method="L-BFGS-B",
        bounds=[(-5, 5), (np.log(1.0), np.log(1e5))],
    )
    m, kappa = expit(res.x[0]), np.exp(res.x[1])
    return m * kappa, (1 - m) * kappa


def posterior_table(teams, A, N, A_eff, N_eff, alpha, beta_, rng):
    pa, pb = alpha + A_eff, beta_ + N_eff - A_eff
    tab = pd.DataFrame(
        {
            "team": teams,
            "n_plants": N.astype(int),
            "n_A": A.astype(int),
            "p_raw": A / N,
            "post_mean": pa / (pa + pb),
            "ci90_lo": beta_dist.ppf(0.05, pa, pb),
            "ci90_hi": beta_dist.ppf(0.95, pa, pb),
        }
    )
    draws = np.clip(rng.beta(pa[:, None], pb[:, None], size=(len(teams), 4000)), 1e-9, 1 - 1e-9)
    H = -(draws * np.log2(draws) + (1 - draws) * np.log2(1 - draws))  # bits
    tab["entropy_mean"] = H.mean(axis=1)
    tab["entropy_lo"] = np.quantile(H, 0.05, axis=1)
    tab["entropy_hi"] = np.quantile(H, 0.95, axis=1)
    return tab


# ---------------------------------------------------------------- Q3: split-half
def split_half_r(tid, a, n, n_teams, rng, shuffle=False):
    """Randomly split each team's matches into two halves; correlate P(A) across teams."""
    if shuffle:
        tid = rng.permutation(tid)
    order = np.lexsort((rng.random(len(tid)), tid))
    t_sorted = tid[order]
    first = np.searchsorted(t_sorted, np.arange(n_teams))
    half = (np.arange(len(tid)) - first[t_sorted]) % 2
    key = t_sorted * 2 + half
    A = np.bincount(key, weights=a[order], minlength=2 * n_teams).reshape(n_teams, 2)
    N = np.bincount(key, weights=n[order], minlength=2 * n_teams).reshape(n_teams, 2)
    ok = (N >= MIN_HALF_PLANTS).all(axis=1)
    if ok.sum() < 5:
        return np.nan
    p = A[ok] / N[ok]
    return np.corrcoef(p[:, 0], p[:, 1])[0, 1]


# ---------------------------------------------------------------- plotting
def forest_plot(tab, p0, path):
    t = tab.sort_values("post_mean").reset_index(drop=True)
    y = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(7.5, 0.28 * len(t) + 1.8))
    ax.hlines(y, t["ci90_lo"], t["ci90_hi"], color="tab:blue", lw=1.5)
    ax.plot(t["post_mean"], y, "o", color="tab:blue", label="shrunk P(A) (90% CI)")
    ax.plot(t["p_raw"], y, "x", color="tab:red", ms=5, label="raw P(A)")
    ax.axvline(p0, ls="--", color="grey", label=f"pooled P(A) = {p0:.2f}")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.team} (n={r.n_plants})" for r in t.itertuples()])
    ax.set_xlim(0, 1)
    ax.set_xlabel("P(plant on A | plant), T side")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main() -> None:
    rng = np.random.default_rng(SEED)
    os.makedirs(f"{OUT_DIR}/figures", exist_ok=True)
    df = load_rounds()
    planted = df[df["planted"]]
    print(
        f"T-side rounds: {len(df)} | plants: {len(planted)} ({len(planted) / len(df):.1%}) "
        f"| pooled P(A|plant): {planted['is_A'].mean():.3f}"
    )
    print("T rounds per team-match (expect ~15 plus overtime):")
    print(df.groupby(["match_id", "team"]).size().describe().round(1))

    u, tid, teams, a, n = prep(df)
    n_teams = len(teams)
    A = np.bincount(tid, weights=a, minlength=n_teams)
    N = np.bincount(tid, weights=n, minlength=n_teams)
    p0 = a.sum() / n.sum()
    print(
        f"\nEligible teams (>= {MIN_PLANTS} plants): {n_teams} | team-match units: {len(u)} "
        f"| plants used: {int(n.sum())} of {len(planted)}"
    )
    print(f"Pooled P(A) over eligible teams: {p0:.3f} | unweighted mean of team P(A): {np.mean(A / N):.3f}")

    # Q1
    het = heterogeneity_test(tid, a, n, n_teams, rng)
    print(
        f"\n[Q1] Between-team heterogeneity (match-level permutation, {N_PERM} perms): "
        f"stat={het['stat']:.1f} vs null mean {het['null_mean']:.1f} "
        f"(95th pct {het['null_p95']:.1f}) -> p = {het['p_value']:.4f}"
    )
    u2, tid2, teams2, a2, n2 = prep(df[~df["is_pistol"]])
    het_np = heterogeneity_test(tid2, a2, n2, len(teams2), rng)
    print(f"     excluding pistol rounds: p = {het_np['p_value']:.4f} ({len(teams2)} teams)")

    # Q2
    phi = dispersion_phi(tid, a, n, n_teams)
    alpha, beta_ = fit_beta_prior(A / phi, N / phi)
    kappa, m = alpha + beta_, alpha / (alpha + beta_)
    sd_between = float(np.sqrt(m * (1 - m) / (kappa + 1)))
    print(f"\n[Q2] quasi-binomial dispersion phi = {phi:.2f} (1 = no within-match clustering)")
    print(f"     prior: mean {m:.3f}, concentration {kappa:.1f}, between-team SD of P(A) ~ {sd_between:.3f}")
    if kappa > 0.99e5:
        print("     NOTE: prior concentration at its upper bound -> no detectable between-team "
              "variation; posteriors collapse to the pooled rate.")
    tab = posterior_table(teams, A, N, A / phi, N / phi, alpha, beta_, rng)
    team_all = df.groupby("team").agg(t_rounds=("planted", "size"), plants=("planted", "sum")).reset_index()
    team_all["plant_rate"] = team_all["plants"] / team_all["t_rounds"]
    tab = tab.merge(team_all, on="team", how="left").sort_values("post_mean")
    tab.to_csv(f"{OUT_DIR}/team_site_preference.csv", index=False)
    forest_plot(tab, p0, f"{OUT_DIR}/figures/team_site_preference_forest.png")

    # Q3
    obs = np.array([split_half_r(tid, a, n, n_teams, rng) for _ in range(N_SPLITS)])
    null = np.array([split_half_r(tid, a, n, n_teams, rng, shuffle=True) for _ in range(N_NULL_SPLITS)])
    obs, null = obs[~np.isnan(obs)], null[~np.isnan(null)]
    split = {}
    if len(obs) and len(null):
        r_obs = float(np.median(obs))
        split = {
            "r_median": r_obs,
            "spearman_brown": float(2 * r_obs / (1 + r_obs)),
            "null_r_mean": float(null.mean()),
            "null_r_p95": float(np.percentile(null, 95)),
            "p_value": float((1 + np.sum(null >= r_obs)) / (1 + len(null))),
        }
        print(
            f"\n[Q3] split-half r (median over {len(obs)} splits) = {r_obs:.3f} "
            f"| null mean {split['null_r_mean']:.3f}, 95th pct {split['null_r_p95']:.3f} "
            f"| p = {split['p_value']:.4f} | Spearman-Brown full-length r = {split['spearman_brown']:.3f}"
        )
    else:
        print("\n[Q3] too few teams with enough plants per half; lower MIN_HALF_PLANTS or MIN_PLANTS.")

    results = {
        "n_teams": int(n_teams),
        "n_units": int(len(u)),
        "plants_used": int(n.sum()),
        "plants_total": int(len(planted)),
        "pooled_p_A": float(planted["is_A"].mean()),
        "heterogeneity": het,
        "heterogeneity_no_pistol": het_np,
        "phi": phi,
        "prior_alpha": float(alpha),
        "prior_beta": float(beta_),
        "between_team_sd": sd_between,
        "split_half": split,
    }
    with open(f"{OUT_DIR}/team_site_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {OUT_DIR}/team_site_preference.csv, team_site_results.json, forest plot.")


if __name__ == "__main__":
    main()
