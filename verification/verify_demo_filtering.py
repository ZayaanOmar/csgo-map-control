import lzma
import json
import glob

ORIGINAL_DEMO = "data/raw_demos/884ea5ad-9174-4475-bc6c-5e031cfe2987.json.xz"
PROCESSED_DEMO = "data/filtered_demos/FaZe_VS_BIG_4bab9a63_42e6_4766_acb6_cae50d575d04.json.xz"

results = {"passed": 0, "failed": 0}

def check(label: str, condition: bool, detail: str = ""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" -- {detail}\n" if detail else "\n"))
    if condition:
        results["passed"] += 1
    else:
        results["failed"] += 1

def load_demo(path: str) -> dict:
    with lzma.open(path, "rt", encoding="utf-8") as f:
        demo: dict = json.load(f)

    return demo

def run_tests():
    # --- / Test 1: Demos have the same keys
    print("=" * 70)
    print("CHECK 1 - DEMO .keys() COMPARISON")
    print("=" * 70)
    check(
        "1.1 - Both demo's have the same top-level keys",
        od_top_keys == pd_top_keys,
        f"Expected Top-Level Keys: {od_top_keys}"
    )

    check(
        "1.2 - Both demo's have the same round-level keys",
        od_round_keys == pd_round_keys,
        f"Expected round-level keys: {od_round_keys}"
    )

    check(
        "1.3 - Both demo's have the same frame-level keys",
        od_frame_keys == pd_frame_keys,
        f"Expected frame-level keys: {od_frame_keys}"
    )

    # --- / Check 2: Map should be de_mirage
    print("=" * 70)
    print("CHECK 2 - MAP NAME SHOULD BE de_mirage")
    print("=" * 70)
    check(
        "2.1 - Processed demo should have mapName = de_mirage",
        pd['mapName'] == "de_mirage",
        f"Map Name: {pd['mapName']}"
    )

    # --- / Check 3: no warmup rounds present
    print("=" * 70)
    print("CHECK 3 - PROCESSED DEMO SHOULD NOT CONTAIN WARMUP ROUNDS")
    print("=" * 70)
    p_rounds = pd['gameRounds']
    containsWarmup = False
    for rnd in p_rounds:
        if rnd['isWarmup']:
            containsWarmup = True

    check(
        "3.1 - Processed demo does not contain warmup rounds",
        not containsWarmup
    )

    # --- / Check 4 - all ticks within correct window
    p_rounds = pd['gameRounds']
    invalidTick = False
    offendingFrame = 0
    offendingRound = 0
    for rnd in p_rounds:
        frames = rnd['frames']

        endTick = rnd['endTick']
        freezeTimeEndTick = rnd['freezeTimeEndTick']
        f_count = 0
        for frame in frames:
            f_count += 1
            tick = frame['tick']
            if  not (freezeTimeEndTick <= tick <= endTick):
                invalidTick = True
                offendingFrame = f_count
                offendingRound = rnd['roundNum']
                break

        if invalidTick:
            break
    check(
        "4.1 - All frame ticks are within correct gameplay window",
        not invalidTick,
        f"Offending Round | Frame: {offendingRound} | {offendingFrame}" if invalidTick else f""
    )

if __name__ == "__main__":

    od = load_demo(ORIGINAL_DEMO)
    pd = load_demo(PROCESSED_DEMO)

    od_top_keys = list(od.keys())
    pd_top_keys = list(pd.keys())

    od_round_keys = list(od['gameRounds'][0].keys())
    pd_round_keys = list(pd['gameRounds'][0].keys())

    od_frame_keys = list(od['gameRounds'][0]['frames'][0].keys())
    pd_frame_keys = list(pd['gameRounds'][0]['frames'][0].keys())

    run_tests()
    print(f"Total Checks Passed: {results['passed']}\nTotal Checks Failed: {results['failed']}")
