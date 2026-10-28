import re
import os
import csv
import json
import glob
import lzma

# --/ Paths relative to root (csgo-map-control)
RAW_DEMOS_PATH = "data/raw_demos"
FILTERED_DEMOS_PATH = "data/filtered_demos"
BOMB_PLANTS_PATH = "outputs/bomb_plants"
ROUND_METADATA_PATH = "outputs/round_metadata"

os.makedirs(FILTERED_DEMOS_PATH, exist_ok=True)
os.makedirs(BOMB_PLANTS_PATH, exist_ok=True)
os.makedirs(ROUND_METADATA_PATH, exist_ok=True)

# --/ Helper functions
def sanitise(name: str) -> str:
    """Replace non-alphanumeric characters with _"""
    return re.sub(r"[^A-Za-z0-9]", "_", name)

def create_match_name(ct: str, t: str, match_id: str) -> str:
    """Create canonical name for a match - ct_VS_t_matchId"""
    return f"{sanitise(ct)}_VS_{sanitise(t)}_{sanitise(match_id)}"

def write_csv(rows: list[dict], path: str, fieldnames: list[str]):
    """Save csv to path"""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

# --/ Demo loading
def load_demos(path: str) -> None:
    """Walk all .json.xz files in path and filter"""
    global num_mirage

    demos = glob.glob(f"{path}/*.json.xz")
    print(f" {len(demos)} demo files found in {path}\n")

    print(f"Found {len(demos)} in {path}")

    for demo_file in demos:
        try:
            with lzma.open(demo_file, mode="rt") as f:
                demo: dict = json.load(f)
        except EOFError:
                print(f"Skipping file due to EOF Error: {demo_file}")
                continue
        except json.JSONDecodeError:
            print(f"Skipping broken JSON structure file: {demo_file}")
            continue

        if demo['mapName'] != "de_mirage":
            continue

        num_mirage += 1
        print(f"[{num_mirage}] Processing: {demo_file}")
        parse_demo(demo)

# --/ Per demo processing
def parse_demo(demo: dict) -> None:
    """
    Filter a demo to de_mirage live-round frames (no warmup rounds, no buy phase, no post round-end frames)
    writes:
        filtered demo json - data/filtered_demos/{match_name}.json.xz
        round metadata csv - outputs/round_metadata/{match_name}.csv
        bomb plants csv - outputs/bomb_plants/{match_name.csv}
    """
    match_id: str = demo["matchId"]
    rounds: list[dict] = demo["gameRounds"]

    filtered_rounds: list[dict] = []
    round_metadata: list[dict] = []
    plant_events: list[dict] = []

    for rnd in rounds:
        # skip warmup rounds
        if rnd.get("isWarmup", False):
            continue

        round_num = rnd["roundNum"]
        freeze_time_end_tick = rnd["freezeTimeEndTick"]
        end_tick = rnd["endTick"]

        # round_metadata row
        round_metadata.append({
            "match_id": match_id,
            "round": round_num,
            "winning_side": rnd["winningSide"],
            "ct_team": rnd.get("ctTeam", ""),
            "t_team": rnd.get("tTeam", ""),
            "round_end_reason": rnd["roundEndReason"],
            "end_tick": end_tick,
            "freeze_time_end_tick": freeze_time_end_tick,
        })

        # bomb plant events
        for event in rnd.get("bombEvents") or []:
            if event["bombAction"] == "plant":
                plant_events.append({
                    "match_id": match_id,
                    "round": round_num,
                    "bomb_site": event["bombSite"],
                    "tick": event["tick"],
                    "seconds": event["seconds"],
                    "planter_name": event["playerName"],
                })

        # Filter to non- buy-phase and non- round-end frames
        kept_frames = [frame for frame in (rnd.get("frames") or []) if freeze_time_end_tick <= frame["tick"] <= end_tick]

        filtered_rounds.append({**rnd, "frames":kept_frames})

    if not filtered_rounds:
        print(f"No non-warmup rounds for match {match_id} - skipping")
        return

    # get names from first competitive round
    ct_team: str = filtered_rounds[0].get("ctTeam") or ""
    t_team: str = filtered_rounds[0].get("tTeam") or ""
    mname: str = create_match_name(ct_team, t_team, match_id)

    # file writes
    filtered_demo = {**demo, "gameRounds": filtered_rounds}
    filtered_demo_path = os.path.join(FILTERED_DEMOS_PATH, f"{mname}.json.xz")
    with lzma.open(filtered_demo_path, "wt", encoding="utf-8") as f:
        json.dump(filtered_demo, f)
    print(f"Filtered demo written to {filtered_demo_path}")

    round_metadata_path = os.path.join(ROUND_METADATA_PATH, f"{mname}.csv")
    write_csv(round_metadata, round_metadata_path, fieldnames=["match_id", "round", "winning_side", "ct_team", "t_team", 
                                                                   "round_end_reason", "end_tick", "freeze_time_end_tick"])
    print(f"Round metadata written to {round_metadata_path}")

    plants_path = os.path.join(BOMB_PLANTS_PATH, f"{mname}.csv")
    write_csv(plant_events, plants_path, fieldnames=["match_id", "round", "bomb_site", "tick", "seconds", "planter_name"])
    print(f"Plant events written to {plants_path} | Number of events: {len(plant_events)}")

if __name__ == "__main__":
    num_mirage = 0
    load_demos(path=RAW_DEMOS_PATH)

    print(f"\n{num_mirage} de_mirage demos filtered")
