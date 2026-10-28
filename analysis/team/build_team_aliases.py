"""Build data/team_aliases.json (raw team name -> canonical name) and print a review report."""
import glob
import json
import os
import re
from difflib import SequenceMatcher
from itertools import combinations

import pandas as pd

OUT_PATH = "data/team_aliases.json"

# Merges that norm() cannot catch on its own. norm() already merges pure case /
# spacing / punctuation variants (compLexity/Complexity, fnatic/Fnatic, TeamVitality/
# Team Vitality, ENTERPRISE/Enterprise, B.I.G/BIG, SINNERS/Sinners, HEROIC/Heroic).
#
# Rules for this dict:
#   * key   = a variant spelling, value = the FINAL canonical name (ONE hop only).
#   * keys are normalised before lookup, so one entry covers every case/spacing
#     variant of that spelling (e.g. "NAVI GGBET" also covers "NAVI GG.BET").
#   * a value must never also be a key (chains are rejected by the assertion below).
# REVIEW THESE: they are guesses from the names alone.
SEED = {
    "Vitality": "Team Vitality",
    "NAVI": "Natus Vincere",
    "NAVI GGBET": "Natus Vincere",       # also covers "NAVI GG.BET"
    "Gambit Esports": "Gambit",
    "OG Esports": "OG",                  # also covers "OGEsports"
    "FURIA Esports": "FURIA",            # also covers "FURIA eSports"
    "BIG Clan": "BIG",
    "Virtus.pro Parimatch": "Virtus.pro",
    "MOUZ": "mousesports",
    "DignitasVIE": "Dignitas",
    "FaZe": "FaZe Clan",
    "forZe": "forZe Esports",
    "Heroic GGBET": "Heroic",            # also covers "HEROIC"
    "Imperial Sportsbet": "Imperial Esports",
    "K23 1xBET": "K23",
    "SINNERS Esports": "Sinners",        # also covers "SINNERS"
    # Abbreviation / rename merges the string heuristics cannot see. CONFIRM:
    "VP Parimatch": "Virtus.pro",
    "EIQ PRG": "Entropiq",
    "CPH Flames": "Copenhagen Flames",
}


def norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


# Normalise both sides so one entry covers all case/spacing variants of a spelling.
SEED_N = {norm(k): norm(v) for k, v in SEED.items()}
assert not set(SEED_N) & set(SEED_N.values()), (
    "SEED has chained entries: point every variant straight at its final name"
)


def main() -> None:
    rm = pd.concat(
        (pd.read_csv(p) for p in glob.glob("outputs/round_metadata/*.csv")),
        ignore_index=True,
    )
    assert rm[["ct_team", "t_team"]].notna().all().all(), "null team names present"
    assert (rm["ct_team"] != rm["t_team"]).all(), "a round has the same team on both sides"
    counts = pd.concat([rm["ct_team"], rm["t_team"]]).value_counts()

    present = {norm(r) for r in counts.index}
    for k in SEED:
        if norm(k) not in present:
            print(f"WARNING: SEED key {k!r} matches no team name in the data (typo?)")

    group = {raw: SEED_N.get(norm(raw), norm(raw)) for raw in counts.index}
    # A group that a SEED entry points at is named exactly as the SEED value (so e.g.
    # "Heroic GGBET" never becomes the display name); other groups use the most
    # frequent spelling.
    seed_names = {norm(v): v for v in SEED.values()}
    canon = {}
    for key in set(group.values()):
        members = [r for r in counts.index if group[r] == key]
        name = seed_names.get(key) or max(members, key=lambda r: counts[r])
        canon.update({r: name for r in members})

    print("== Merged groups ==")
    for name in sorted(set(canon.values())):
        members = sorted(r for r in canon if canon[r] == name)
        if len(members) > 1:
            print(f"  {name:25s} <- {members}")

    keys = {group[r]: canon[r] for r in canon}  # one entry per group
    print("\n== Possible unmerged duplicates (review aid only) ==")
    for (ka, a), (kb, b) in combinations(sorted(keys.items()), 2):
        short, long_ = sorted((ka, kb), key=len)
        if (
            long_.startswith(short)
            or (len(short) >= 5 and short in long_)
            or SequenceMatcher(None, ka, kb).ratio() >= 0.85
        ):
            print(f"  {a!r}  vs  {b!r}")

    t_counts = rm["t_team"].map(canon).value_counts()
    print(f"\n== {len(set(canon.values()))} canonical teams from {len(canon)} raw names ==")
    for name, n in t_counts.items():
        print(f"  {name:28s} T-side rounds: {n}")
    print("\nT-side rounds per team (describe):\n", t_counts.describe())
    print("teams with >= 40 T-side rounds:", int((t_counts >= 40).sum()))

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(canon, f, indent=2, sort_keys=True)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
