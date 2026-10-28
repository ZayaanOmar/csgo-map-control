import networkx as nx
from awpy.data import NAV, NAV_GRAPHS

map_name = "de_mirage"
g = NAV_GRAPHS[map_name]
components = list(nx.strongly_connected_components(g))
components.sort(key=len, reverse=True)

main_component = components[0]
orphan_areas = set().union(*components[1:])

print(f"Main component size: {len(main_component)}")
print(f"Orphaned areas: {len(orphan_areas)}\n")

for area_id in orphan_areas:
    info = NAV[map_name][area_id]
    print(f"  areaId={area_id:>4}  areaName={info['areaName']!r}  "
          f"centroid_z={(info['northWestZ'] + info['southEastZ'])/2:.1f}")

for area_id in [295, 833, 363]:
    print(f"areaId={area_id} ({NAV[map_name][area_id]['areaName']})")
    print(f"    in_edges: {list(g.in_edges(area_id))}")
    print(f"    out_edges: {list(g.out_edges(area_id))}")
