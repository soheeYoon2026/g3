# Geometry preparation pipeline

One command takes a supplier's STEP and produces everything the pipeline can:
a healed STEP for the CAD engineer, a patched mesh for the solver path, the
projected frontal area, pictures, and a list of the decisions only an engineer
can make.

```
.venv/bin/python scripts/prepare_geometry.py \
    --in CAS-A.stp --out var/runs/cas-a \
    --seal-below 900 \
    --close-near=-41,-910,76,450 --close-near=2689,-905,62,450 \
    --wrap
```

Note the `=` in `--close-near=-41,...`: a value starting with a minus sign is
otherwise read as an option.

## Stages and outputs

| stage | what it does | writes |
|---|---|---|
| `cad` | read STEP, diagnose, sew in stages (1 → 5 → 10.5 mm on a car) | `cad.json` |
| `heal` | close holes under the sealing size as B-rep caps on the boundary's own edges; verify every patch; write STEP | `healed.stp`, `heal.json` |
| `intent` | list what was left open, with position, size and reason | `intent.md` |
| `mesh` | tessellate with consistent winding, stitch tessellation seams, patch the rest (flat by default; pocket rims stay flat) | `mesh.stl`, `mesh_full.stl`, `mesh.json`, `mesh.txt` |
| `area` | projected frontal area of the full car, rasterised union | `frontal_area.txt` |
| `render` | four-view pictures with the open boundaries drawn on | `render_step.png`, `render_mesh.png` |
| `wrap` | (`--wrap`) CGAL alpha wrap of the mesh; hollow until the large openings are closed | `wrap.stl`, `wrap.json`, `wrap.txt` |

`summary.json` records which stages ran, how long, and the key numbers. A stage
that fails is recorded and the rest still run. `log.txt` is the console output.

A mesh input (`.stl`, `.obj`, `.ply`) skips `cad`, `heal` and `intent`.

## Options that need a human

- **`--seal-below`** — holes up to this size are closed without asking. It has to
  sit below the smallest opening the flow must pass through (grille slot, duct,
  cooling inlet), or those get sealed too. Default is a sixth of the diagonal,
  which is a guess; on CAS-A 900 mm was used deliberately to include the wheels.
- **`--close-near X,Y,Z,R`** — close boundaries near this point even when there is
  surface behind them. This is how a closed rim is requested. The report labels
  them as simplifications.
- **`--no-mirror`** — the model is already a full car.

What the pipeline refuses to decide is written to `intent.md`: openings larger
than the sealing size (underbody, cabin band), overlapping styling panels (glass
sitting on the body, which needs the supplier's trimmed surfaces), and the
occasional tool failure. Those are the questions for the supplier.

## Individual tools

Every stage is also a standalone script under `scripts/`:

- `heal_step.py` — the B-rep stage alone, with `--list-only` to just see the holes
- `fair_mesh.py` — the mesh stage alone
- `frontal_area.py` — area of any STEP/STL, with `--mirror` and `--ground-z`
- `render_geometry.py` — pictures, with `--focus-hole N` / `--focus-point X,Y,Z`
- `check_topology.py` — intersections (slow, `--intersections`), hidden-face
  classification (refuses on a leaking body), orientation, curvature meshing
- `seal_geometry.py` — the wrap tier
- `audit_*.py`, `measure_*.py`, `sweep_*.py` — the measurements the design
  decisions rest on; `VERSION2_PLAN.md` explains each

## Where the numbers come from

Every threshold in `aox_g3/brep.py` and `aox_g3/fair.py` was set from a measured
gap between two populations on real geometry, and the comment next to each
constant says which measurement. When a new model breaks one, re-measure before
moving it.
