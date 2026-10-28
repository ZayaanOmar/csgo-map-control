"""
area_assignment.py
------------------
Give every one of de_mirage's 903 nav tiles an area label, including the 72
tiles whose awpy ``areaName`` is blank ("").

Rules (design option 1 from the discussion)
    1. Tiles that already have a name keep it.
    2. The two confirmed blank clusters are handled by hand:
         ApartmentsRamp = 873-880 (8 tiles) is hand-labelled with its OWN label
           (the awpy "Apartments" tiles are the B-side apps, a different place;
           to merge later, edit MERGE_MAP below).
         The CT stairs (511, 512, 525-558, 561-566; 42 tiles) are SPLIT between
           CTSpawn and TicketBooth: each tile takes whichever of the two is
           closer (3D gap between the tile's box and the nearest tile of that
           name; ties go to CTSpawn and are flagged).
    3. Every remaining blank tile ("scattered singleton") takes the label that
       wins a vote among its geometric neighbours:
         * neighbour = XY rectangle gap <= tolerance AND z-intervals within
           Z_GAP_MAX units (so tiles stacked above/below each other do NOT
           count - e.g. Catwalk over the Tunnel nook);
         * vote weight = length of shared edge (corner touch counts 1);
         * tolerance starts at TOUCH_TOL (2.0, same as the graph patch) and
           widens along TOL_LADDER only for tiles with no valid neighbour;
         * labels propagate: a tile whose only neighbours are other unresolved
           singletons is resolved in a later round, once they are labelled;
         * last resort: nearest labelled tile by 3D centroid distance.
       Every decision is recorded (method, tolerance, vote share, runner-up)
       so it can be audited.

Outputs
    data/de_mirage_area_assignment.csv   one row per tile (all 903)
    + a printed review table of the 22 singletons and any ambiguous votes.

Optional audit (--patched-graph PATH)
    The graph patch used an XY-only gap test. This lists edges from the 72
    formerly-blank tiles to tiles whose z-range is far away (> --zflag units),
    i.e. edges that may connect different floors.

Usage (from project root)
    python pipeline/area_assignment.py
    python pipeline/area_assignment.py --patched-graph data/de_mirage_patched_graph_v3.pkl

Downstream:  from area_assignment import load_assignment
             labels, formerly_blank = load_assignment("data/de_mirage_area_assignment.csv")
"""

import argparse
import csv
import math
import os
import pickle
from collections import Counter, defaultdict

MAP_NAME = "de_mirage"

# --- hand-labelled clusters (confirmed by map knowledge) --------------------
HAND_CLUSTERS = {
    "ApartmentsRamp": list(range(873, 881)),
}

# Blank tiles assigned to the CLOSER of several existing areas (3D box gap).
SPLIT_CLUSTERS = {
    "CTStairs": dict(
        ids=[511, 512] + list(range(525, 559)) + list(range(561, 567)),
        candidates=["CTSpawn", "TicketBooth"],
    ),
}

# Optional merges applied AFTER assignment, e.g. {"ApartmentsRamp": "Apartments"}
MERGE_MAP: dict[str, str] = {}

# --- neighbour-vote settings -------------------------------------------------
TOUCH_TOL = 2.0                                   # same tolerance as the graph patch
TOL_LADDER = [2.0, 10.0, 25.0, 50.0, 100.0, 200.0]
Z_GAP_MAX = 20.0                                  # max vertical gap between z-intervals
AMBIGUOUS_SHARE = 0.60                            # flag if winner has < 60% of vote


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def _box(area: dict) -> dict:
    xs = (area["northWestX"], area["southEastX"])
    ys = (area["northWestY"], area["southEastY"])
    zs = (area["northWestZ"], area["southEastZ"])
    return {
        "x0": min(xs), "x1": max(xs),
        "y0": min(ys), "y1": max(ys),
        "z0": min(zs), "z1": max(zs),
    }


def xy_gap(a: dict, b: dict) -> float:
    """Axis-aligned rectangle gap distance (0 if touching/overlapping)."""
    dx = max(b["x0"] - a["x1"], a["x0"] - b["x1"], 0.0)
    dy = max(b["y0"] - a["y1"], a["y0"] - b["y1"], 0.0)
    return math.hypot(dx, dy)


def z_gap(a: dict, b: dict) -> float:
    """Vertical gap between the two tiles' z-intervals (0 if they overlap)."""
    return max(b["z0"] - a["z1"], a["z0"] - b["z1"], 0.0)


def box_gap_3d(a: dict, b: dict) -> float:
    """Shortest 3D distance between two axis-aligned boxes (0 if they intersect)."""
    return math.hypot(xy_gap(a, b), z_gap(a, b))


def contact_len(a: dict, b: dict) -> float:
    """Length of the shared edge (0 for a corner touch)."""
    ox = min(a["x1"], b["x1"]) - max(a["x0"], b["x0"])
    oy = min(a["y1"], b["y1"]) - max(a["y0"], b["y0"])
    return max(ox, oy, 0.0)


def _centre(b: dict) -> tuple:
    return ((b["x0"] + b["x1"]) / 2, (b["y0"] + b["y1"]) / 2, (b["z0"] + b["z1"]) / 2)


# --------------------------------------------------------------------------- #
# Assignment
# --------------------------------------------------------------------------- #
def assign_areas(nav: dict, hand_clusters: dict = None, z_gap_max: float = Z_GAP_MAX,
                 tol_ladder: list = None, split_clusters: dict = None):
    """Return (labels, info, warnings).

    nav   : {area_id: {"areaName", "northWestX", ... "southEastZ"}}  (awpy NAV[map])
    labels: {area_id: label}
    info  : {area_id: dict(original, method, tol, round, share, runner_up,
                           n_neighbours, n_z_rejected)}
    """
    hand_clusters = HAND_CLUSTERS if hand_clusters is None else hand_clusters
    split_clusters = SPLIT_CLUSTERS if split_clusters is None else split_clusters
    tol_ladder = TOL_LADDER if tol_ladder is None else tol_ladder

    boxes = {int(i): _box(a) for i, a in nav.items()}
    orig = {int(i): (a["areaName"] if isinstance(a["areaName"], str) else "")
            for i, a in nav.items()}

    labels: dict[int, str] = {}
    info: dict[int, dict] = {}
    warnings: list[str] = []

    # diagnostic: XY-touching tiles that the z-test rejects (floors stacked above/below)
    zrej: dict[int, int] = {}
    for i, n in orig.items():
        if n == "":
            zrej[i] = sum(1 for j, bj in boxes.items()
                          if j != i and xy_gap(boxes[i], bj) <= TOUCH_TOL
                          and z_gap(boxes[i], bj) > z_gap_max)

    def _blank_info(i, **kw):
        d = dict(original=orig[i], method="", tol=0.0, round=0, share=1.0,
                 runner_up="", n_neighbours=0, n_z_rejected=zrej.get(i, 0))
        d.update(kw)
        return d

    # 1. named tiles keep their name
    for i, n in orig.items():
        if n != "":
            labels[i] = n
            info[i] = _blank_info(i, method="original")

    # 2. hand-labelled clusters
    for label, ids in hand_clusters.items():
        for i in ids:
            if i not in orig:
                warnings.append(f"hand-labelled id {i} ({label}) is not in NAV")
            elif orig[i] != "":
                warnings.append(f"hand-labelled id {i} ({label}) already has name "
                                f"{orig[i]!r}; keeping the original")
            else:
                labels[i] = label
                info[i] = _blank_info(i, method="hand_cluster")

    # 2b. split clusters: closer of the candidate areas (named tiles only)
    for cname, spec in split_clusters.items():
        cands = spec["candidates"]
        cand_ids = {c: [j for j, n in orig.items() if n == c] for c in cands}
        for c in cands:
            if not cand_ids[c]:
                warnings.append(f"split cluster {cname}: candidate area {c!r} has no tiles")
        for i in spec["ids"]:
            if i not in orig:
                warnings.append(f"split-cluster id {i} ({cname}) is not in NAV")
                continue
            if orig[i] != "":
                warnings.append(f"split-cluster id {i} ({cname}) already has name "
                                f"{orig[i]!r}; keeping the original")
                continue
            dist = {c: min((box_gap_3d(boxes[i], boxes[j]) for j in cand_ids[c]),
                           default=float("inf")) for c in cands}
            ranked = sorted(cands, key=lambda c: (dist[c], cands.index(c)))
            best, other = ranked[0], ranked[1]
            tot = dist[best] + dist[other]
            labels[i] = best
            info[i] = _blank_info(i, method="split_nearest", runner_up=other,
                                  share=(dist[other] / tot) if tot > 0 else 0.5,
                                  tol=dist[best], n_neighbours=len(cands))
            info[i]["dist"] = dist
            if dist[best] == dist[other]:
                warnings.append(f"id {i}: tie between {best} and {other} at distance "
                                f"{dist[best]:.1f}; assigned {best}")

    unresolved = sorted(i for i in orig if i not in labels)

    # 3. vote, widening the tolerance only when a tile has no valid neighbour
    for tol in tol_ladder:
        rnd = 0
        progressed = True
        while progressed and unresolved:
            progressed = False
            rnd += 1
            new_labels = {}
            for i in unresolved:
                votes: dict[str, float] = defaultdict(float)
                min_gap: dict[str, float] = {}
                n_nb = 0
                for j, lab in labels.items():      # snapshot: labels not mutated in this round
                    g = xy_gap(boxes[i], boxes[j])
                    if g > tol or z_gap(boxes[i], boxes[j]) > z_gap_max:
                        continue
                    w = max(contact_len(boxes[i], boxes[j]), 1.0) if g <= TOUCH_TOL else 1.0 / g
                    votes[lab] += w
                    min_gap[lab] = min(min_gap.get(lab, g), g)
                    n_nb += 1
                if not votes:
                    continue
                ranked = sorted(votes, key=lambda L: (-votes[L], min_gap[L], L))
                total = sum(votes.values())
                new_labels[i] = (ranked[0], dict(
                    original="", method="touch_vote" if tol <= TOUCH_TOL else f"expanded_{tol:g}",
                    tol=tol, round=rnd, share=votes[ranked[0]] / total,
                    runner_up=ranked[1] if len(ranked) > 1 else "",
                    n_neighbours=n_nb, n_z_rejected=zrej.get(i, 0)))
            for i, (lab, d) in new_labels.items():
                labels[i] = lab
                info[i] = d
            if new_labels:
                progressed = True
                unresolved = [i for i in unresolved if i not in new_labels]

    # 4. last resort: nearest labelled tile by 3D centroid distance
    for i in unresolved:
        ci = _centre(boxes[i])
        best = min((j for j in labels), key=lambda j: math.dist(ci, _centre(boxes[j])))
        labels[i] = labels[best]
        info[i] = dict(original="", method="centroid_fallback", tol=float("inf"), round=0,
                       share=0.0, runner_up="", n_neighbours=0,
                       n_z_rejected=zrej.get(i, 0))
        warnings.append(f"id {i}: no valid neighbour at any tolerance; used nearest "
                        f"centroid ({best} -> {labels[best]!r})")

    # 5. optional merges
    if MERGE_MAP:
        labels = {i: MERGE_MAP.get(L, L) for i, L in labels.items()}

    # carry original names (blank stays blank) into info
    for i in info:
        info[i]["original"] = orig[i]
    return labels, info, warnings


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
FIELDS = ["area_id", "original_areaName", "assigned_area", "method", "tol_used",
          "round", "vote_share", "runner_up", "n_neighbours", "n_z_rejected"]


def write_assignment(path: str, labels: dict, info: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        for i in sorted(labels):
            d = info[i]
            w.writerow([i, d["original"], labels[i], d["method"],
                        "" if d["method"] in ("original", "hand_cluster") else d["tol"],  # split_nearest: distance to winner
                        d["round"], round(d["share"], 3), d["runner_up"],
                        d["n_neighbours"], d["n_z_rejected"]])


def load_assignment(path: str):
    """Return (labels {id: label}, formerly_blank {ids whose original name was blank})."""
    labels, formerly_blank = {}, set()
    with open(path, newline="", encoding="utf8") as f:
        for row in csv.DictReader(f):
            i = int(row["area_id"])
            labels[i] = row["assigned_area"]
            if row["original_areaName"] == "":
                formerly_blank.add(i)
    return labels, formerly_blank


# --------------------------------------------------------------------------- #
# Optional audit of the patched graph
# --------------------------------------------------------------------------- #
def audit_patched_edges(nav: dict, formerly_blank: set, graph_path: str, z_flag: float,
                        out_csv: str) -> None:
    with open(graph_path, "rb") as f:
        G = pickle.load(f)
    boxes = {int(i): _box(a) for i, a in nav.items()}
    pairs = set()
    for u in formerly_blank:
        if u not in G:
            continue
        for v in set(G.successors(u)) | set(G.predecessors(u)):
            pairs.add((min(u, v), max(u, v)))
    rows = []
    for u, v in sorted(pairs):
        zg = z_gap(boxes[u], boxes[v])
        if zg > z_flag:
            rows.append((u, v, round(zg, 1), round(xy_gap(boxes[u], boxes[v]), 1)))
    rows.sort(key=lambda r: -r[2])
    print(f"\nPATCHED-GRAPH Z AUDIT: {len(pairs)} distinct edges touch a formerly-blank tile; "
          f"{len(rows)} link tiles whose z-ranges are > {z_flag:g} units apart")
    for r in rows[:25]:
        print(f"  {r[0]:>4} <-> {r[1]:<4} z-gap {r[2]:>6}  xy-gap {r[3]}")
    if len(rows) > 25:
        print(f"  ... and {len(rows) - 25} more (see {out_csv})")
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(["tile_a", "tile_b", "z_gap", "xy_gap"])
        w.writerows(rows)


# --------------------------------------------------------------------------- #
# Report + CLI
# --------------------------------------------------------------------------- #
def print_report(labels: dict, info: dict, warnings: list) -> None:
    blank = [i for i in info if info[i]["original"] == ""]
    print(f"\n{len(labels)} tiles labelled; {len(blank)} were blank")
    for w in warnings:
        print(f"  WARNING: {w}")

    print("\nFormerly-blank tiles by assigned label:")
    for lab, n in sorted(Counter(labels[i] for i in blank).items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {lab:<18} {n:>3}")

    split = sorted(i for i in blank if info[i]["method"] == "split_nearest")
    if split:
        print(f"\nSplit-cluster tiles ({len(split)}): closer of the candidate areas "
              f"(distances = 3D gap to nearest tile of that name)")
        print(f"  {'id':>4}  {'assigned':<12} {'dist':>7}  {'other':<12} {'dist':>7}")
        for i in split:
            d = info[i]
            print(f"  {i:>4}  {labels[i]:<12} {d['dist'][labels[i]]:>7.1f}  "
                  f"{d['runner_up']:<12} {d['dist'][d['runner_up']]:>7.1f}")

    singles = sorted(i for i in blank
                     if info[i]["method"] not in ("hand_cluster", "split_nearest"))
    print(f"\nScattered singletons ({len(singles)}):")
    print(f"  {'id':>4}  {'assigned':<14} {'method':<14} {'rnd':>3} {'share':>6} "
          f"{'runner-up':<14} {'n_nb':>4} {'z-rej':>5}")
    for i in singles:
        d = info[i]
        flag = "  <-- ambiguous" if (d["runner_up"] and d["share"] < AMBIGUOUS_SHARE) else ""
        if d["method"] == "centroid_fallback":
            flag = "  <-- fallback"
        print(f"  {i:>4}  {labels[i]:<14} {d['method']:<14} {d['round']:>3} {d['share']:>6.2f} "
              f"{d['runner_up']:<14} {d['n_neighbours']:>4} {d['n_z_rejected']:>5}{flag}")

    n_z = sum(1 for i in blank if info[i]["n_z_rejected"] > 0)
    print(f"\n{n_z} of {len(blank)} formerly-blank tiles XY-touch at least one tile on a different "
          f"floor (rejected by the z test). Use --patched-graph to see whether the patch linked them.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/de_mirage_area_assignment.csv")
    ap.add_argument("--z-gap", type=float, default=Z_GAP_MAX)
    ap.add_argument("--patched-graph", default=None, help="optional: audit patched edges for z jumps")
    ap.add_argument("--zflag", type=float, default=40.0)
    ap.add_argument("--audit-out", default="outputs/area_name_map/patched_edges_z_audit.csv")
    args = ap.parse_args()

    from awpy.data import NAV
    nav = NAV[MAP_NAME]

    labels, info, warnings = assign_areas(nav, z_gap_max=args.z_gap)
    write_assignment(args.out, labels, info)
    print_report(labels, info, warnings)
    print(f"\nWrote {args.out}")

    if args.patched_graph:
        blank = {i for i in info if info[i]["original"] == ""}
        audit_patched_edges(nav, blank, args.patched_graph, args.zflag, args.audit_out)


if __name__ == "__main__":
    main()
