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
| `wrap` | (`--wrap`) CGAL alpha wrap of the mesh; hollow until the large openings are closed; a watertight input is left alone unless `--force-wrap` | `wrap.stl`, `wrap.json`, `wrap.txt` |
| `local` | (`--local-wrap`, with `--keep-openings-above`) re-wrap only the closed openings at half the keep size and splice them into the coarse wrap; the coarse decisions stand elsewhere | `wrap_local.stl`, `local.txt`, `local_wrap.json` |
| `smooth` | (`--smooth-seams`) remesh and smooth only the seams the wrap added (plate edges, tube junctions); everything else pinned | `wrap_smooth.stl`, `smooth.txt` |

`summary.json` records which stages ran, how long, and the key numbers. A stage
that fails is recorded and the rest still run. `log.txt` is the console output.

A mesh input (`.stl`, `.obj`, `.ply`) skips `cad`, `heal` and `intent`.

## Letting the pipeline propose its own parameters

```
.venv/bin/python scripts/propose_parameters.py --in CAS-A.stp --out params.json
.venv/bin/python scripts/prepare_geometry.py --in CAS-A.stp --out var/runs/cas-a --auto
```

`--auto` (or `--params params.json` from the first command) measures the model
and fills in what a person otherwise types: the unit, whether it is a half model
and where the symmetry plane is, the sewing ladder (a single-pass sweep picks the
start where invalid faces are lowest, a cumulative sweep from there picks how far
the tolerance may climb), the sealing size (the widest gap in the hole-size
distribution), and closed-rim points (round loops, low, near either end). Every
proposal comes with its reasoning in `params.json`, and any explicit flag
overrides it. It does not decide intent; the questions it cannot answer are
listed under `questions`.

## Options that need a human

- **`--seal-below`** — holes up to this size are closed without asking. It has to
  sit below the smallest opening the flow must pass through (grille slot, duct,
  cooling inlet), or those get sealed too. Default is a sixth of the diagonal,
  which is a guess; on CAS-A 900 mm was used deliberately to include the wheels.
- **`--close-near X,Y,Z,R`** — close boundaries near this point even when there is
  surface behind them. This is how a closed rim is requested. The report labels
  them as simplifications.
- **`--no-mirror`** — the model is already a full car.
- **`--keep-openings-above MM`** — the smallest opening the flow must pass through
  (wing slot, duct, grille). The wrap closes every gap narrower than about twice
  its alpha, so this sets alpha to half that size; `wrap_closed.txt` lists what
  the wrap closed anyway, with the gap each patch bridged. On the formula car,
  13 mm kept all four wing slots and the floor wedge open (alpha 6 mm, 854k
  triangles) while 10 mm alpha had glued the slot lips and the wing mounts.
- **`--local-wrap`** — with the option above: keep the coarse alpha everywhere and
  re-wrap only the closed openings at half the keep size (contact prevention).
  On the formula car: 407k triangles instead of 854k, same openings kept, hollow
  tubes still filled. Needed when the openings to keep are small (grille slots)
  and one fine alpha everywhere would cost tens of millions of triangles. With
  `--smooth-seams` the remesh is sized from the coarse alpha in this mode (599k
  triangles on the formula car; 1.01M if sized from the fine one).

What the pipeline refuses to decide is written to `intent.md`: openings larger
than the sealing size (underbody, cabin band), overlapping styling panels (glass
sitting on the body, which needs the supplier's trimmed surfaces), and the
occasional tool failure. Those are the questions for the supplier.

## Runbook

Environment: Python 3.9, `pip install cadquery-ocp==7.7.0 trimesh scipy numpy shapely
networkx pillow rtree cgal manifold3d`. Geometry lives under `var/` (ignored) and nothing
that contains geometry goes anywhere else while the repository is public.

Three standard runs:

```
# A. everything: STEP -> healed.stp + intent.md + mesh.stl + frontal area + renders (+ wrap)
prepare_geometry.py --in X.stp --out var/runs/x --seal-below 900 --close-near=X,Y,Z,R --wrap
# B. let the model propose its parameters
propose_parameters.py --in X.stp --out params.json        # look only
prepare_geometry.py --in X.stp --out var/runs/x --auto   # or --params params.json
# C. watertight under a declared flat-floor assumption
flat_floor_wrap.py --in var/runs/x/mesh_full.stl --out var/runs/x-assumed --alpha-div 360
add_floor_step.py --in var/runs/x/healed.stp --floor var/runs/x-assumed/floor.stl \
                  --out var/runs/x-assumed/healed_half_floor.stp --no-mirror
```

Read `summary.json` for stage status and numbers (`free_boundaries_measured` is
measured; `holes_left` is bookkeeping), `heal.json` for every hole's verdict and
reason, `intent.md` for the questions, `params.json` for proposals with reasons.
A bounding box that differs from the input means a patch escaped.

Common failures: a `--close-near` value starting with `-` must be joined with
`=`; a wrap stage that says `unchanged` means the input was already watertight
(closed multi-body STL) — add `--force-wrap` to wrap it anyway; a wrap that comes back `hollow` means the large openings are still open
(intent, or run C); a skipped wrap tier means `cgal` is not installed; slow STEP
opening means the mirror was written as a copy — write the half with
`--no-mirror` and mirror in CAD.

Regression after any change: `heal_step.py --in var/cad/visibility_test.stp`
must report 0 holes, closed, 0.204 m³; the clean `Cv10.STEP` must pass through
with 1,083 faces unchanged.

## Individual tools

Every stage is also a standalone script under `scripts/`:

- `heal_step.py` — the B-rep stage alone, with `--list-only` to just see the holes
- `fair_mesh.py` — the mesh stage alone
- `frontal_area.py` — area of any STEP/STL, with `--mirror` and `--ground-z`
- `render_geometry.py` — pictures, with `--focus-hole N` / `--focus-point X,Y,Z`
- `check_topology.py` — intersections (slow, `--intersections`), hidden-face
  classification (refuses on a leaking body), orientation, curvature meshing
- `seal_geometry.py` — the wrap tier; `wrap_once.py --alpha-div --offset` for one wrap
- `list_closed_openings.py` — what a wrap closed (patch centre, size, bridged gap);
  `overlay_sections.py` — reference vs candidate section overlays to look at them
- `local_wrap.py` — coarse wrap + fine local re-wrap + boolean splice (manifold3d)
- `smooth_wrap.py` — seam smoothing after a wrap (`--remesh T --smooth taubin`);
  `measure_wrap_roughness.py` reports the dihedral angles of the seams vs the rest
- `audit_*.py`, `measure_*.py`, `sweep_*.py` — the measurements the design
  decisions rest on; `VERSION2_PLAN.md` explains each

## Where the numbers come from

Every threshold in `aox_g3/brep.py` and `aox_g3/fair.py` was set from a measured
gap between two populations on real geometry, and the comment next to each
constant says which measurement. When a new model breaks one, re-measure before
moving it.
