import json
from awpy.data import NAV

bombsite_nodes = {"A": [], "B": []}
for area_id, area in NAV["de_mirage"].items():
    if area["areaName"] == "BombsiteA":
        bombsite_nodes["A"].append(int(area_id))
    elif area["areaName"] == "BombsiteB":
        bombsite_nodes["B"].append(int(area_id))

with open("data/de_mirage_bombsite_nodes.json", "w", encoding="utf-8") as f:
    json.dump(bombsite_nodes, f, indent=2)