# G3 v6 data audit

## Smoke report

Source: `data/smoke_report/solver_smoke_all_tests_final_report.csv`

| Solver | Rows | Succeeded | Failed | Report rows with Cd | Independent test IDs |
|---|---:|---:|---:|---:|---:|
| G1 | 12 | 8 | 4 | 8 | 8 |
| G2 | 19 | 6 | 13 | 1 | 8 |
| G4 | 13 | 5 | 8 | 1 | 8 |
| Total | 44 | 19 | 25 | 10 | 8 |

The report summary omitted coefficients for several successful runs. S3
evidence recovered them from G2 `history.csv` and G4 `*_results.json` files.
Retries and DRAG/LIFT reruns of the same geometry were not counted as new
independent test shapes.

## G2 smoke materialization

Successful G2 jobs yielded FINAL and `RBF_DSN_*` field states. Exact duplicate
`surface_flow.vtu` artifacts across objective reruns were removed.

| Test | Materialized field cases | Coefficient-supervised | Field-only | Final integrated Cd / Cl |
|---|---:|---:|---:|---:|
| test002 | 15 | 15 | 0 | 0.8888 / 0.1731 |
| test006 | 12 | 12 | 0 | 0.2694 / 0.2036 |
| test011 | 7 | 7 | 0 | 0.2568 / -0.0095 |
| test018 | 2 | 0 | 2 | 0.1258 / 0.0433 (rejected as coefficient labels) |
| Total | 36 | 34 | 2 | — |

`test018` has only 3,704 surface points. Its Cp/velocity fields remain in the
shared field dataset, but its Cd/Cl labels do not contribute to coefficient
loss because the v6 threshold is 5,000 points.

## Combined G2 v6 set

| Item | Count |
|---|---:|
| Existing v5 field cases | 113 |
| Added smoke G2 field cases | 36 |
| Total field cases | 149 |
| Accepted G2 Cd/Cl supervision | 87 |
| Field-only / rejected coefficient labels | 62 |

The previously suspicious high-Cd Ahmed labels are among the rejected rows:
the 342-point surface meshes produced Cd near 1.0–1.17, while sufficiently
resolved versions of the same geometry produced Cd near 0.23–0.26.

## Other coefficient domains

| Expert | Training evidence | Cases | Policy |
|---|---|---:|---|
| `g2_su2_clean` | Quality-gated G2 Cd <= 0.5 | 75 | Production default |
| `g2_su2_high_drag` | Quality-gated G2 Cd > 0.5 | 12 | Explicit selection; limited domain |
| `g1_openfoam` | Existing G1 Cd 27 + G1 Cl 11 + smoke G1 fields 2 | 40 | Explicit selection only |
| `g4_lbm` | Successful smoke G4 optimized STL + `cd_best` | 5 | Experimental; strict OOD gate |

Four additional G1 smoke jobs had zero `UMean` in their final exported
inlet/outlet VTP state, so they were not fabricated into surface-field labels.
Their test geometry families are still represented through successful G2 or
G4 artifacts. Failed solver runs remain failure/OOD evidence and are never
used as regression targets.
