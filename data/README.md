# data/

## Included (static artefacts produced by `pipeline/`)

| File | Produced by | Contents |
|---|---|---|
| `de_mirage_patched_graph_v3.pkl` | `pipeline/patch_de_mirage_graph_v3.py` | Patched de_mirage nav graph (networkx DiGraph): 903 nodes, 3,505 edges |
| `de_mirage_apsp_dict.pkl` | `pipeline/all_pairs_shortest_path_gen.py` | 903×903 directed geodesic distance matrix (`inf` = unreachable, expected from the 3-SCC structure) plus area↔index maps |
| `de_mirage_bombsite_nodes.json` | `pipeline/get_bombsite_nodes.py` | Nav-area IDs for bombsites A and B |
| `de_mirage_area_assignment.csv` | `pipeline/area_assignment.py` | All 903 nodes resolved to 27 area names (blank-name tiles filled) |
| `de_mirage_bridge_log_v3.json` | `pipeline/patch_de_mirage_graph_v3.py` | Edges added by the patch (orphan bridges and internal links); read by `visualisation/plot_patched_nav_mesh.py` |
| `team_aliases.json` | `analysis/team/build_team_aliases.py` | Raw team name → canonical team name |
| `filtered_demos_manifest.sha256` | `sha256sum` over `data/filtered_demos/` | SHA-256 of each of the 278 filtered de_mirage demos |

The pickles were written with Python 3.11 / networkx 3.6.1 (see `requirements.txt`).

## Not included: ESTA demos

The match data come from the ESTA dataset (Xenopoulos et al.), <https://github.com/pnxenopoulos/esta>; see that
repository for its licence. They are not redistributed here because of size.

1. Download the ESTA demos and put all `.json.xz` files (LAN and online) directly in `data/raw_demos/`
   (one flat folder).
2. Run `python pipeline/filter_demos.py` to copy the de_mirage matches that
   pass the filters into `data/filtered_demos/`.
3. Check the subset matches the one used in the report:

```bash
cd data/filtered_demos && sha256sum -c ../filtered_demos_manifest.sha256
```
