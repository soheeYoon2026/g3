# G3 version2 — Connection Map and Development Plan

Written: 2026-08-24
Reference document: [docs/G3_HANDOVER_2026-08-24.md](docs/G3_HANDOVER_2026-08-24.md) (a copy of `G3_작업_전체_정리.md` from the Downloads folder)

## 1. What this folder is

A development folder created by cloning `../g3` (soheeYoon2026/g3, branch
`fix/unify-surface-preprocessing`) with its full history and branching `version2`.

- Four uncommitted pieces of recent work were carried over from the original g3:
  `README.md`, `scripts/evaluate_domino_v3.py` (modified),
  `LIBRARIES.md`, `scripts/evaluate_pointnet_domino_split.py` (new).
- The origin still points at `git@github.com:soheeYoon2026/g3.git`.
  Pushing sends work to the `version2` branch of the original repository.
- `data/`, `var/`, and `predictions/` are gitignored, so they start empty.
  Pull v1 artefacts from `../g3/data` (441M) and `../g3/var` (593M) as needed.
  The small checkpoints and evaluation artefacts under `models/` (8.7M) were copied over.

## 2. Connection map — verified locally on 2026-08-24

### 2.1 Serving inference path (already merged into the local develop branch)

```
aox-next (develop)                      aox-django (develop)                    this repository
g3-preview-panel.tsx        ──POST──▶   app/g3_inference/views.py   ──POST──▶  aox_g3/service.py (FastAPI)
g3-three-viewer.tsx                     /api/v1/g3/inferences                  /v1/infer  (Bearer token)
                                        · checks team admin permission          · runs infer_fields
                                        · looks up the GPU service url/token   · returns Cd/Cl + OOD + preview PNG
                                          from S3 _private/g3/inference-service.json
```

- **`integration/aox-django/src/app/g3_inference/` was confirmed byte-identical
  by diff to the code merged into aox-django develop.** This repository is the
  source of truth for the Django integration.
- Django setting names: `G3_INFERENCE_URL`, `G3_INFERENCE_TOKEN`,
  `G3_INFERENCE_TIMEOUT_SECONDS`, `G3_INFERENCE_CONFIG_BUCKET`,
  `G3_INFERENCE_CONFIG_KEY` (`integration/aox-django/src/config/settings/base.py`).
- Service-side environment variables: `G3_MODEL_PATH`, `G3_COEFFICIENT_EXPERT`,
  `G3_API_TOKEN` (or `G3_API_TOKEN_FILE`), `G3_MAX_UPLOAD_BYTES`.

### 2.2 Full Workbench G3 (temp/g3-workbench-transfer)

Of the four repository commits named in the handover document, what was located
and secured locally:

| Repository | Documented commit | Local status |
|---|---|---|
| aox-next | `21c24cc3` | **fetched from origin/temp/g3-workbench-transfer** (hash matches) |
| aox-workbench | `8e3fda6` | **fetched from origin/temp/g3-workbench-transfer** (hash matches) |
| aox-django-backend | `ce9215e` | the local aox-django's remote is `ADRO-DEVEL/aox-django` and has no temp branch. `aox-django-backend` appears to be a **different repository**, not cloned — needs confirmation |
| aox-next-admin | `003d40c` | the repository itself is not present locally — not cloned |

G3 files the transfer branch adds over develop/canary:

- aox-next: `g3-surface-viewer.tsx`, `g3-profile-viewer.tsx`,
  `g3-surface-deform.ts(+spec)`, `g3-profile-deform.ts(+spec)`,
  `g3-profile-controls.tsx`, `g3-point-preview.worker.ts`,
  `g3-original-stl-export.ts` — the entire 72-control-point surface editing,
  symmetry, undo, and STL export UI.
- aox-workbench: `src/features/mesh/ui/g3/G3Workbench.tsx`,
  `G3SurfacePointEditor.tsx` (29 files, +1,467 lines).

In short, **local develop only has "STL upload → Cd/Cl preview"; the control-point
editing, ΔCd comparison, and recommendation UI live in the fetched transfer branch.**

### 2.3 Not available locally (be aware when working)

- The production checkpoint `decoder-30epoch.pt` and DoMINO experiment artefacts —
  G3_TEST `/home/ubuntu/g3-v2/var/` (document §11). Recovering uncommitted G3_TEST
  code is still outstanding.
- Project/Celery async structure and the admin-screen backend — `aox-django-backend`,
  `aox-next-admin` (not cloned).
- Training source data — S3 `aoxlabs-stage-static`, `aoxlabs-prod-static`,
  and the archival `s3://strain-bucket-adro`.

## 3. Training pipeline (this repository's DoMINO v3 chain)

```
run_scheduled_collection.py      midnight collection entry point (collection-only safe mode)
  └ collect_s3_training_data.py  collect AOX events → build v2 NPZ → update manifest
prepare_domino_su2_v3.py         SU2 surface results → flow-aligned DoMINO data (Mach/AoA preserved)
build_domino_su2_v3.py           audit-based full v3 dataset build + physics validation
build_quality_gated_manifest.py  merge Cd/Cl label quality gates
split_domino_v3_groups.py        deterministic geometry-group split
create_group_holdout.py          group-isolated train/holdout manifests
finetune_domino_v3.py            DoMINO fine-tune validated against SU2 Cd (with dedup)
evaluate_domino_v3.py            evaluate pretrained/fine-tuned checkpoints  ← being modified
evaluate_pointnet_domino_split.py evaluate PointNet on the same split  ← being written
```

See [docs/G3_PREPROCESSING_V2_RUNBOOK.md](docs/G3_PREPROCESSING_V2_RUNBOOK.md) for
the preprocessing v2 validation procedure, and
[docs/G3_NIGHTLY_CANARY.md](docs/G3_NIGHTLY_CANARY.md) for nightly collection and
canary operations.

## 4. Local environment (this machine)

- `.venv`: Python 3.9 with `--system-site-packages`. **torch 2.8.0+cu128, CUDA
  available (local GPU).** `fastapi`, `uvicorn`, `python-multipart`, `pytest`, and
  `eval_type_backport` were installed — absent from the original g3 venv — so that
  **the inference service can be started locally**. (`eval_type_backport` is needed
  to evaluate `service.py`'s `str | None` syntax on Python 3.9; unnecessary on a
  3.10+ environment such as G3_TEST.)
- Tests: `.venv/bin/python -m pytest -q` → 27 passing (2026-08-24)
- Service `/health` startup confirmed (2026-08-24, model_ready with `g3_field_cdcl_v4.pt`)
- Local service startup (smoke):

  ```bash
  cd /home/adro1234/2026/SU2_work/g3-version2
  G3_API_TOKEN=devtoken G3_MODEL_PATH=models/g3_field_cdcl_v4.pt \
    .venv/bin/uvicorn aox_g3.service:app --port 8005
  curl -s localhost:8005/health
  ```

  Note: the local checkpoints under `models/` (cdcl_v4, g1_v2) are preprocessing-v1
  artefacts and may be incompatible with v2 inference code (the runbook states this
  explicitly: v1 checkpoints fail deliberately under v2 code). For a reliable smoke
  test, build a fresh tiny checkpoint as in runbook §3, or pull a current checkpoint
  from G3_TEST/S3.
- Django integration test locally: give aox-django `G3_INFERENCE_URL=http://127.0.0.1:8005`
  and `G3_INFERENCE_TOKEN=devtoken` and it connects directly, bypassing the S3 config lookup.

## 5. version2 development plan (based on handover document §10)

| # | Task | Where |
|---|---|---|
| 1 | NISMO re-verification — pin the STL hash, units, axes, flow conditions, and G2 step between the AI input and the G2 original | this folder (verification script) + comparison against G2 results |
| 2 | Safely commit the current checkpoint and preprocessing code | this folder's `version2` branch + G3_TEST code recovery |
| 3 | G2 runs for inward/outward deformations of the same control point at several magnitudes | reuse the transfer branch's deformation logic (`g3-surface-deform.ts`) + G2 pipeline |
| 4 | Add original/variant pairs for vehicle families beyond the GT-R | collection pipeline (§3 chain) |
| 5 | Unified evaluation: absolute Cd MAE + ΔCd MAE + direction accuracy | extend `evaluate_domino_v3.py` (continuing the in-progress edit) |
| 6 | Promote only challengers that win on a fully family-isolated test | `create_group_holdout.py` + canary runbook |

Startable immediately: #5 (the evaluation script edit is already in progress), #1
(local SU2 results in the `optimization_rans/steady_oneram6` family can be compared),
and the local portion of #2. Requiring G3_TEST access: recovering the server code
for #2, and obtaining the current decoder-30epoch.pt.

## 6. Progress log

### 2026-08-24 — #5 unified evaluation metrics, #1 re-verification tooling (local portion done)

**#5: added `aox_g3/eval_metrics.py`.** Computes absolute Cd/Cl MAE + Spearman +
**ΔCd/ΔCl MAE, direction accuracy, and rank correlation** in one summary. Pairs come
from two sources:

- Group-derived: all combinations of cases sharing a `group_id` in the split (default)
- Explicit pairs: `--pairs pairs.json`, format
  `{"pairs": [{"baseline": <run>, "variant": <run>}]}` — this is the mode for
  evaluating G3_TEST's deformation experiments (27/39/57/84 cases)

Both `evaluate_domino_v3.py` and `evaluate_pointnet_domino_split.py` use this module,
so DoMINO/PointNet comparisons come out under identical metric definitions.
`--direction-tolerance` excludes pairs with small |ΔCd| from the direction score.

Caution: within-group pairs in the local `su2_labels_v3` split (4 train, 16 test)
are near-duplicate geometries whose true ΔCd tops out around 0.0006. For direction
accuracy to mean anything, either supply a tolerance or evaluate against an explicit
pair manifest from the deformation experiments.

**#1: added `scripts/verify_case_identity.py`.** Produces an identity report between
the AI input STL and the G2 case: STL sha256 + v3 geometry_digest (computed the same
way as the manifest), bounding box/axis/unit determination, +X projected frontal area,
cfg flow-condition comparison, and **per-step CD/CL tracking through the history files
plus a search for the step that produced a given Cd** (`--find-cd`).

Verified on run_14: the label `su2_cd` is the value at the last step (404) of
`history.csv`, and convergence within tolerance began around step 320. So the current
label convention is "the last row of history.csv"
(`prepare_domino_su2_v3.read_su2_coefficients`). NISMO re-verification runs on
G3_TEST like this:

```bash
python scripts/verify_case_identity.py \
  --stl <NISMO AI input.stl> --case-dir <NISMO G2 run directory> \
  --find-cd 0.3207 --strict --out nismo_identity.json
```

Tests: 40 passing, including `tests/test_eval_metrics.py` and
`tests/test_verify_case_identity.py`.

**Outstanding:** the actual NISMO run for #1 (needs G3_TEST material), server code
recovery for #2, and everything from #3 onward (G2 runs for deformations, additional
vehicle families, challenger gates).

### 2026-08-25 — P1 final verdict: G2↔LES ΔCd cross-validation complete on 6 pairs

Protocol (the final form after seven rounds of trial and error,
[docs/LES_PROTOCOL_NOTES.md](LES_PROTOCOL_NOTES.md)):
the watertight surface G2 solved (drivaer_N) · 3-level standard · 18k samples ·
free flight · u_inf 0.10 · cs_l2 0.16 · **refine1/2 grids frozen from the baseline**
(env `LES_FIX_REFINE_JSON`, patched into the deployed copy of amr_les.py) ·
rear_wide alone on a pair-internal grid (the micro-triangle bbox pin destabilised
that geometry specifically — a colleague's warning proved right).
cd_std 0.0003-0.0007 per run, four-digit reproducibility between baseline hosts.

| Pair | G2 ΔCd | LES ΔCd | Direction |
|---|---:|---:|---|
| front_narrow_5mm | −0.0164 | −0.0097 | ✅ |
| roof_raise_5mm | −0.0046 | −0.0140 | ✅ |
| rear_narrow_10mm | −0.0019 | −0.0350 | ✅ |
| rear_narrow_25mm | +0.0083 | +0.0143 | ✅ |
| roof_lower_10mm | +0.0095 | −0.0262 | ❌ (sign reproduced in 3 configurations) |
| rear_wide_25mm | +0.0088 | −0.0029 | ❌ (\|LES\| small; v5 reproduced −0.0026) |

**Verdict: direction agreement 4/6 (67%), magnitude rank correlation 0.03 (uncorrelated).**

### 2026-08-26 — challenger-mixed-v1 training and gate results (partial improvement, promotion withheld)

Training: SU2 54 + DrivAerML 71 mixed, decoder mode, 30 epochs, per-epoch validation
best-state, LR 1e-4. The first attempt wiped out with NaN at LR 2e-4 (no checkpoint
saved) — after adding a guard, the retrain identified **run_1006 alone as the source
of the non-finite loss** (skipped; a candidate for exclusion next round). Checkpoint
at `var/domino-automotive-runs/challenger-mixed-v1.pt` (G3_TEST).

| Gate | Current decoder-30epoch | challenger-mixed-v1 | Verdict |
|---|---|---|---|
| 63 unseen shapes, MAE / Spearman | 0.2516 / −0.03 | 0.2580 / **+0.11** | ❌ cliff unresolved |
| 71 training-exposed cases, MAE | 0.0334 | **0.0234** | ✅ improved |
| SU2 family-isolated test, 8 cases, MAE | 0.0219 | 0.0256 | ≈ equivalent |
| gtr 6-pair ΔCd direction (vs G2) | 4/6 | **5/6** | ✅ improved |
| gtr 6-pair ΔCd MAE | 0.0068 | **0.0051** | ✅ improved |
| gtr 6-pair ΔCd Spearman | 0.49 | **0.60** | ✅ improved |

Reading: adding 71 DrivAerML cases **clearly improved the ΔCd (recommendation) axis**
but did not close the unseen-arbitrary-shape generalization gap (the gate-2 cliff) —
DrivAer morphs are "one more family," not coverage of the arbitrary distribution of
AOX uploads. **Withheld** for failing the promotion gate (improvement on unseen).
Note: because the challenger is trained on G2 labels, it aligns with G2 on the
disputed pairs (vs LES 4/6 → 3/6).

Next levers: ① add WindsorML and AhmedML families ② rebuild a fair gate of
"complete vehicles" by filtering components and abnormal shapes out of the 63 unseen
cases ③ encoder-tail training mode ④ if these fail, settle the product positioning
(OOD gate + within-family only).

### 2026-08-26 — Fair-gate re-verdict: "zero ranking ability" was an artefact of gate contamination

Reclassifying by manifest flow conditions shows that of the 265 reaudit cases,
**182 meet the complete-vehicle conditions** (30 m/s, incompressible, ref_area
1.2-4.5, AoA 0), and the existing "63 unseen" gate was **heavily contaminated with
transonic wings (run_1: Mach 0.84, 285 m/s), components, and special shapes**.
The fair gate on 15 unseen complete cars:

| Model | MAE | Median relative error | Spearman |
|---|---:|---:|---:|
| pretrained | 0.3453 | 116% | −0.37 |
| **decoder-30epoch (current)** | **0.1042** | **25%** | **0.86** |
| challenger-mixed-v1 | 0.1792 | 74% | 0.61 |

Three-line re-verdict:
1. **The current model ranks previously unseen complete cars well (0.86)** — absolute
   values carry ~25% error (consistent with NISMO). The gate-2 conclusion of "zero
   ranking ability" was an over-diagnosis produced by non-car contamination.
2. **The challenger regresses against the current model on the fair gate** — the
   DrivAerML mixture dilutes AOX-car specialisation (traded for the ΔCd axis
   improvement). Promotion stays withheld.
3. Product implications: (a) the "relative comparison/ranking of new cars" feature is
   already usable with the current model (b) absolute Cd still needs an OOD gate
   (c) **classifying and warning about component/non-car uploads at the input stage is
   cheaper and more urgent than improving the model** — half the performance problem
   was an input-distribution problem.

Next-round design (reflecting the dilution lesson): don't mix DrivAerML into the same
batch; use **a curriculum (DrivAerML pretraining → SU2-only finish)** or domain-weighted
sampling, and exclude run_1006. Freeze the gate as these "15 complete cases" (plus
newly collected G2 cases).

### 2026-08-26 (2nd) — WindsorML new-family probe + upload gate

**Upload gate implemented** (`aox_g3/upload_gate.py`, commit 58fda1a): classifies
full_car/component/non_car from geometric rules alone, 16/16 on real data, with an
`upload_gate` warning field in the `service.py` response. YOLO (Ultralytics) was
excluded as AGPL, and external vision APIs were excluded over customer design leakage —
a second pass on ambiguous cases is planned with CLIP (MIT).

**WindsorML probe (12 cases, car-size ×4 scale)** — two issues found and fixed along
the way: Windsor fields are stored as non-dimensional coefficients (cpavg/cf*avg), so
a ×0.5U² conversion was added; and zero-area cells from decimation caused NaN in
DoMINO's area division (removal logic folded into ingest). It was also confirmed that
because the DoMINO wrapper's coordinate normalization box is fixed to DrivAer's
dimensions, model-scale inputs must be scaled up ×4.

| Model | Windsor MAE | Windsor Spearman |
|---|---:|---:|
| pretrained | 0.3245 | −0.17 |
| **decoder-30epoch** | **0.0629** | −0.10 |
| challenger-mixed | 0.1472 | −0.31 |

Overall reading: ① even on a completely new family, the current model's **absolute Cd
holds up at around 0.06** ② but ranking fine variations within a new family is
**beyond all three models** (Spearman ≈ 0) — "within-family ranking" is an ability
that only appears after fine-tuning on that family's data (consistent with the ΔCd
experiments) ③ the challenger regressed on three consecutive new axes → **withholding
is now final**, and the next attempt is the curriculum approach.

1. **G2 ΔCd labels are conditionally valid as "direction labels"** — 2/3 agreement.
   They diverge on separation-dominated deformations such as roof and rear-widening
   changes, and the magnitudes cannot be trusted.
2. **decoder-30epoch re-verdict: 4/6 against G2, 4/6 against LES** — the model's errors
   are not solely the labels' fault (rear_narrow_10 is wrong against both). But on
   roof_lower the model sides with LES (suggesting the G2 label is wrong), and on
   rear_wide it sides with G2.
3. Practical conclusion: G2 remains usable as a ΔCd training label (direction-weighted,
   magnitude down-weighted), and **disputed pairs go to LES for adjudication** — cases
   like the 2 of these 6 where solvers disagree should be excluded from training or
   replaced with the LES value. Keep the G2 verification gate in the recommendation UX.
4. The LES campaign infrastructure (fixed-grid A/B, pair-internal grids, cs env) is
   deployed on the host in a reusable form (~/gtr-les/, results runs_v5-v7 preserved).

### 2026-08-24 (2nd) — G3_TEST access: gate 1 verdict + server material recovery

**Gate 1 (NISMO convention verification) — the error is genuine model error, not a
convention problem:**

- The cited value "0.3207 per G2" **could not be found in any G2 artefact**.
  Measured NISMO-family full-car G2 values: run_60 (raw GT-R) su2_cd `0.3025`,
  gtr-smooth baseline label `0.30790` (history final = rbf source), VTU integration
  of the same geometry `0.31232` (+1.4%), and integrated values of the high-drag
  variants `0.3202-0.3209`.
- Convention differences quantified: ref_area AI `3.1201` (bbox frontal) vs G2 cfg
  `3.104241` (**0.5%**); history label vs VTU integration (**+1.4%**); G2 convergence
  noise tail std `1.5e-5` (negligible). Together a few percent at most —
  **the ~50% gap between AI `0.4723` and G2 `0.30-0.31` is out-of-distribution model error.**
- Mesh identity: the AI input `gtr-nismo-latest.stl` (419,428 tri) and the campaign
  baseline STL (511,548 tri) are **different meshes**. Future comparisons must use the
  same mesh.
- G2 scale convention discovered: each job's `geometry_scale.json` — source in mm,
  normalized to a 5.000 m maximum dimension in solver space
  (`solver_units_per_meter 0.99782`).

**Bonus — a real ΔCd pair campaign was found:** the `gtr-smooth` G2 campaign, 7 cases
(baseline + 6 variants, ΔCd −0.0164 to +0.0095, over 100× the noise) → frozen as
[benchmarks/gtr_smooth_pairs.json](benchmarks/gtr_smooth_pairs.json).
This is the first real data for `--pairs` evaluation and the existing half of the P1
campaign. Seven variant STLs that were uploaded but never computed are waiting in S3
(the P1 expansion).
LOO fine-tune experiment record: training on 5 cases took holdout Cd MAE 0.0119 → 0.0082.

**#2 server material recovery:** 47 uncommitted files from G3_TEST `~/g3-v2` (the
serving stack, the ΔCd builders, systemd units), a patch of modifications to tracked
files (626 lines), and small experiment records were recovered into
`recovered/g3-test-20260824/`. The server's modified `evaluate_domino_v3.py` is on a
**separate lineage** from the local version and must be compared before merging
(`recovered/g3-test-20260824/g3v2-tracked-modifications.patch`).

**Next execution order:** ① run the current decoder-30epoch over the 7 gtr-smooth cases
to measure `--pairs` ΔCd direction accuracy (server GPU, ~1.5 s per case) ② submit G2
for the 7 uncomputed variants (P1 expansion) ③ LES pilot P0/P1.

### 2026-08-24 (3rd) — First measurement of ΔCd direction accuracy (① complete)

Evaluated the 6 gtr-smooth pairs on the G3_TEST GPU with the unified metrics
([benchmarks/results/gtr_smooth_decoder30_20260824.jsonl](benchmarks/results/gtr_smooth_decoder30_20260824.jsonl),
tolerance 0.001):

| Model | Absolute Cd MAE | Cd Spearman | ΔCd MAE | **Direction accuracy** | ΔCd Spearman |
|---|---:|---:|---:|---:|---:|
| pretrained (NVIDIA) | 0.2374 | −0.14 | 0.0132 | 3/6 = 50% | −0.20 |
| decoder-30epoch | 0.0121 | 0.43 | 0.0068 | **4/6 = 67%** | 0.49 |

Per pair: accuracy improves with deformation size — front_narrow_5mm (true Δ −0.0164)
was nearly reproduced at −0.0146, and rear_wide/rear_narrow_25 got the direction right.
Two failures: roof_lower_10mm (true +0.0095 → pred −0.0021) and rear_narrow_10mm (the
smallest deformation, −0.0019 → +0.0101). The predicted Cd range (0.012) is half the
actual (0.026) — a tendency to under-predict change.

Interpretation and limits: ① 6 pairs is a small sample, so 67% is barely a straw —
it needs the 13-pair expansion (submitting the 7 uncomputed variants) to mean anything
② the evaluated variants are unseen cases created (8/20) after training (8/13,
reaudit-v1), but the GT-R family itself was exposed in training via run_60 — this is
not out-of-family generalization ③ it looks like an improvement over the earlier
"50% direction" report, but that was not the same set, so a direct comparison is invalid.

### 2026-08-26 (3rd) — Curriculum v1 verdict: no promotion, but a per-use conclusion

Stage A (DrivAerML 71, run_1006 restored, 15 epochs) → Stage B (SU2 54 only, 30 epochs,
continuing from the base checkpoint). Stage B SU2 validation 0.0109 / isolated test 0.0358.

| Model | Fair-15 MAE/rank | gtr ΔCd direction/MAE | Windsor MAE/rank |
|---|---|---|---|
| Current decoder-30ep | **0.104 / +0.86** | 4/6 / 0.0068 | **0.063** / −0.10 |
| mixed-v1 | 0.179 / +0.61 | **5/6 / 0.0051** | 0.147 / −0.31 |
| Curriculum v1 | 0.150 / **+0.83** | 4/6 / 0.0066 | 0.129 / −0.09 |

Verdict: ① the curriculum recovers most of the mixture's dilution of specialisation
(rank 0.83) but does not surpass the current model — **no promotion on the absolute
Cd/generalization axis; keep the current model** ② mixed-v1 is the only one to beat the
current model on the ΔCd axis (5/6, 0.0051) — preserved as a **recommendation-only
expert candidate** (consistent with the v6 multi-expert structure; the recommendation
pipeline uses only ΔCd) ③ external-family pretraining does not buy unseen-case
improvement at this scale (SU2 54) — the real lever for the fair gate is
**expanding collection of AOX-domain complete cars** (nightly collector + upload gate
integration: automatically admit only new G2 results judged full_car).

### 2026-08-26 (4th) — Recommendation-only expert wiring (server service)

Added **per-task model routing** to
`recovered/g3-test-20260824/scripts/domino_stl_service.py` (= the G3_TEST deployment):

- `G3_DOMINO_RECOMMEND_CHECKPOINT` env makes only the recommendation path use a
  different checkpoint (unset behaves exactly as before — verified as a no-change
  deployment with `recommend_expert_split:false`)
- A separate resident engine (`_load_recommend_engine`) keeps both models loaded
- **Re-anchoring** (`_reanchor_to_absolute_model`): the expert's baseline Cd diverged
  from the preview (/v1/infer, serving model), which surfaced **the same car showing
  two different Cd values** → absolutes are anchored to the serving model, the expert's
  ΔCd is kept, and each candidate's absolute Cd is recomputed as `anchor + ΔCd`.
  The originals are preserved as `expert_cd/expert_cl`.
- `/health` exposes `checkpoint`, `recommend_checkpoint`, and `recommend_expert_split`;
  the recommendation response exposes `absolute_model` and `delta_model`.

Verification (NISMO, shadow 8011 vs 8012 on identical code): baseline unified at 0.4723
(the expert's own 0.3608 preserved), candidate absolute Cd = baseline + ΔCd consistent,
the top two candidates shared by both models (point_00_out/point_03_out), differing only
at third place.

**Not yet applied in production** — adding one env line
(`G3_DOMINO_RECOMMEND_CHECKPOINT=... challenger-mixed-v1.pt`) to the systemd unit and
restarting switches it over, and deleting that line rolls it back immediately. Having
the frontend display `absolute_model`/`delta_model` would avoid user confusion.

### 2026-08-26 (5th) — Production switch + collector gate integration

**① Recommendation expert applied in production.** systemd drop-in
`/etc/systemd/system/g3-domino-inference.service.d/recommend-expert.conf`
(one env line) → live verification: `absolute_model: decoder-30epoch.pt`,
`delta_model: challenger-mixed-v1.pt`, baseline 0.4723 (same as the preview),
candidate Cd = baseline + ΔCd. **Rollback = delete that drop-in file and restart.**

**② Collector ↔ upload gate integration.** Added `classify_case(conditions, geometry)`
to `aox_g3/upload_gate.py` (which also judges the flow regime: speed 5-80 m/s,
Mach ≤ 0.3, |AoA| ≤ 5°). The server collector `build_domino_s3_v4.py` records a
`shape_class` for every accepted case and leaves the distribution in the manifest
summary. Passing `--require-car-case` rejects non-car cases outright (the default only
tags them — the data is kept and filtered at split time).

Retroactive tagging of the existing 265 cases: **car_case 163 / component 50 /
non_car_shape 21 / unsure 29 / off_regime 2**.

**Standard gate frozen** (`benchmarks/unseen_car_gate.json`, 13 cases =
unseen ∩ car_case):

| Model | MAE | Median relative | Spearman |
|---|---:|---:|---:|
| **Current decoder-30ep** | **0.0881** | **24%** | **+0.87** |
| mixed-v1 | 0.1639 | 75% | +0.60 |
| Curriculum v1 | 0.1469 | 48% | +0.86 |

Challenger promotion decisions are now reproducible from the 13 cases in this file.

### 2026-08-26 (6th) — Gate exposed in production inference + frontal-area definition fixed

Added `upload_gate` to the production `/v1/infer` response (domino_stl_service.py).
The Django proxy passes the payload through unchanged, so the frontend only needs to
read this field to display a warning (`verdict`, `reasons`, `features`).

**Frontal-area definition bug fixed**: the initial implementation used `Σ|n·A|/2`
(the projected sum of every face), so wheels, underbody, and interior panels overlapped
and inflated NISMO to 6.18 m² (actual 3.12 m²), producing a false `unsure`. Replaced
with the **bbox frontal area (width × height)** — NISMO comes out at 3.12 m², matching
the G2 REF_AREA, and the verdict normalises to `full_car`. Real data 16/16 correct
(11 cars, 5 components), and `unsure` in the retroactive tagging fell from 29 to 2.

Re-tagging results (265 cases): **car_case 190 / component 40 / non_car_shape 31 /
unsure 2 / off_regime 2**.

**Standard gate v2 frozen** (`benchmarks/unseen_car_gate.json`, 15 cases):

| Model | MAE | Median relative | Spearman |
|---|---:|---:|---:|
| **Current decoder-30ep** | **0.0929** | **24%** | **+0.91** |
| mixed-v1 | 0.1642 | 54% | +0.65 |
| Curriculum v1 | 0.1362 | 45% | +0.90 |

For the frontend (aox-next) it is enough to show a caution badge next to the
coefficients when `uploadGate.verdict !== 'full_car'` (mind the DRF camelCase conversion).

### 2026-08-26 (7th) — New 7-pair LES campaign: 6 of 7 unusable, methodology lessons secured

An attempt to expand to 13 pairs by labelling the 7 uncomputed variants with LES. The
geometries are Workbench uploads that never went through G2, so there is no watertight
surface: ① vertex welding (1,554 open edges remaining) ② union bbox pinning
③ computed at standard/18k/free/cs 0.16.

| Variant | LES ΔCd | S/N | Verdict |
|---|---:|---:|---|
| roof_lower_20mm | **−0.0101** | **2.9** | ✅ usable |
| front_wide_10mm | −0.0039 | 1.1 | ❌ |
| front_narrow_10mm | −0.0025 | 0.7 | ❌ |
| front_wide_20mm | −0.0024 | 0.7 | ❌ |
| front_narrow_20mm | −0.0008 | 0.2 | ❌ |
| cabin_narrow_5mm | −0.0003 | 0.1 | ❌ |
| roof_lower_5mm | +0.0002 | 0.1 | ❌ |

**Of 13 pairs only 7 are usable for direction judgement** (the original 6 plus
roof_lower_20mm). The 6 new pairs were buried in noise, and the front-end deformations
even showed narrow and wide both reducing drag — a contradiction that makes the absence
of signal plain.

Two layers of cause: ① the open shell gives a plateau sd **4×** that of watertight
geometry (0.0035 vs 0.0005) ② bbox pinning unifies the grid but also **pins the
reference area**, erasing the frontal-area reduction of narrow/wide deformations — only
the roof deformations, which are independent of frontal area, kept their signal
(monotone in magnitude too: 5 mm 0 → 20 mm −0.0101).

**Additional gain**: roof_lower became a third independent LES result supporting the
opposite direction from G2 (G2 +0.0095 vs LES −0.0262/−0.0101). G2's roof-deformation
ΔCd labels are safer excluded from training or replaced with LES values.

**Design criteria for the next campaign** (measured):
1. Deformations for ΔCd labels must exceed **3× the noise** (|ΔCd| ≥ 0.0015 on
   watertight geometry, ≥ 0.010 on an open shell).
2. **Run G2 first to obtain the watertight surface (VTU), then feed LES** — reversing
   the order quadruples the noise.
3. State the reference-area handling explicitly for deformations that change frontal
   area (pinning measures shape effect only).

### 2026-08-26 (8th) — Surface-field approach: the ΔCp signal is real, and training doubles it

**Motivation**: a single ΔCd buries small deformations in CFD noise (the v8/v9 failures),
so the first question was whether the surface field retains the signal. On the 6
gtr-smooth pairs, **CFD ΔCp inside the deformed patch is 1.27-2.33× the far field**
(6/6) — the local signal survives even on G2's coarse mesh. Each pair yields
**40,000 cells** of observation rather than one scalar, so 6 pairs support statistics.

**New evaluator** `scripts/evaluate_surface_delta.py` (commit 29c84b2): compares the
model's predicted ΔCp field against the CFD ΔCp field (patch correlation and signal
ratio, with the far field as a control).

**New training** `challenger-paired-v1`: paired batches (baseline and variant passed
together) + a **ΔCp difference loss** + up to 4× weight on cells where the CFD ΔCp is
large. G2 re-meshes each variant so cell counts differ; nearest-centre mapping handles
this (the first attempt skipped all 6 pairs for exactly this reason).

| Metric | Current decoder-30ep | **paired-v1** |
|---|---:|---:|
| Patch ΔCp correlation (median, training pairs) | +0.30 | **+0.64** |
| Model/CFD signal ratio (front_narrow) | 1.93 / 2.33 | **2.26** / 2.33 |
| Family-isolated test Cd MAE | 0.0219 | **0.0198** |
| Standard gate 15 MAE / rank | **0.0929 / +0.91** | 0.0993 / +0.82 |
| Windsor MAE | **0.0629** | 0.0643 |
| gtr ΔCd direction / MAE | **5/6 / 0.0051** | 4/6 / 0.0058 |

**Verdict**: the local surface-field signal clearly improved (+0.30 → +0.64, signal
ratio approaching CFD's) and the out-of-training-pair metric (family-isolated test) got
better too. But **on the unseen gate it is equal or slightly behind** (0.0993/+0.82 vs
0.0929/+0.91) and integrated ΔCd direction fell to 4/6 — that is, **the model learned to
capture the local pressure field, and that did not translate into better integrated Cd.**
Promotion withheld, but the direction is proven.

**Next**: ① optimise the integrated Cd loss jointly while preserving the ΔCp correlation
(multi-objective) ② add training pairs to reduce room for overfitting ③ freeze the
evaluation metric as the pair "ΔCp correlation + ΔCd direction" for judging future
challengers.

### 2026-08-27 — Multi-objective (ΔCp + integrated Cd) training: no promotion, campaign one closed

`challenger-multi-v1` = paired-v1's ΔCp pair loss + a **differentiable integrated Cd
loss** (an unbiased estimate scaling the step's 4,096-cell sample by N/K, `sampled_cd()`),
cd_weight 0.5.

| Model | Gate15 MAE/rank | Windsor MAE | ΔCd direction/MAE | Patch ΔCp correlation |
|---|---|---|---|---|
| **Current decoder-30ep** | **0.0929 / +0.91** | **0.0629** | **5/6 / 0.0051** | +0.30 |
| Curriculum v1 | 0.1362 / +0.90 | 0.1288 | 4/6 / 0.0066 | — |
| paired-v1 | 0.0993 / +0.82 | 0.0643 | 4/6 / 0.0058 | **+0.64** |
| multi-v1 | 0.1409 / +0.83 | 0.1598 | 5/6 / 0.0055 | +0.48 |

**Verdict**: multi-v1 recovered ΔCd direction to 5/6 but **badly degraded unseen
generalization** (Windsor 0.0629 → 0.1598, gate 0.0929 → 0.1409). Training validation
was the best ever recorded (Cd MAE 0.0054) while the family-isolated test came out at
0.0352 — textbook overfitting. The integrated Cd loss pulled the model toward the
absolute values of the training family.

**Campaign one conclusion**: all four challengers (mixed, curriculum, paired, multi)
failed to beat the current model on the standard gate. **The serving model remains
decoder-30epoch**, with only the recommendation path using the mixed-v1 expert (applied
in production).

**What was gained**: ① proof that surface-field ΔCp is a measurable signal source for
small deformations (CFD signal ratio 1.3-2.3×, 6/6) ② that signal can be doubled by
training (+0.30 → +0.64) ③ but **local accuracy does not propagate to integrated Cd or
generalization**, and adding the integral term directly causes overfitting.

**Top priority next**: increase the training data itself — 109 shapes across 78 families
were confirmed on hand (local-only scan), and the G2 → watertight surface → LES
verification pipeline now exists. Today's conclusion is that loss design has nothing
more to give.

---

### 2026-08-27 — Moving to the official NVIDIA pipeline, and finding a units mismatch

Following the observation that "this doesn't feel like doing it the way the paper does,"
the hand-written decoder fine-tune was abandoned in favour of **PhysicsNeMo's official
`train.py` (Hydra config, resuming from `nvidia/domino_drivaerml`)**. After clearing
eight environment problems (example-library version mismatch, cuml cu13/cu12 mixing,
CUDA header contamination, epochs < checkpoint epoch, sample counts exceeding case size,
bounding-box mismatch, and others) and expanding EBS from 100 to 200 GB, the first
training run completed.

**And the result was garbage**: validation loss fell 44%, from 153,402 to 86,439, while
predicted Cd on the gate came out as **−47, +105, −24** (MAE 94.5, ΔCd direction 2/6).

**Cause — units mismatch.** The pressure field described by the pretrained checkpoint's
`scaling_factors.pkl` is **non-dimensional** (mean −0.2104, std 0.2474, range −2.4373 to
0.8568), while `pMeanTrim` in our G2 `boundary_N.vtp` is in **Pascals** (±1000-3600).
Feeding Pa against Cp-scale statistics **does not error** — training runs, the loss
falls, checkpoints save. It only surfaces at evaluation.

**The normalization constant was established by measurement, not inference**
(`measure_field_normalization.py`). Solving per case for the constant N that makes the
evaluator's surface integral reproduce the recorded `su2_cd` converges, across 12 cases
from 27.8 to 285.7 m/s, on **N = ρU² = 2q with 4% spread**. So the training target is
**`p_Pa / (ρU²)`** (= Cp/2, the DrivAerML convention). The shear columns are
**already non-dimensional** (ours ±0.03 vs DrivAerML's ±0.018) and must not be touched.

**Diagnostic signal**: with correct normalization the training loss sits around **1.0**.
Tens of thousands means a units problem, not a hard problem. After the fix the field
range is −3.27 to +3.27.

**The conversion path was verified with a control first.**
`convert_mdlus_checkpoint.py` is an **identity** on the pretrained checkpoint (0 of 381
tensors changed) and asserts the architecture matches. So a bad gate score cannot be
blamed on the conversion. This also produced the first measurement of the pretrained
model's own score — a **floor**.

| Model | Gate15 MAE / rank | Windsor MAE | gtr Cd MAE | ΔCd direction |
|---|---|---|---|---|
| pretrained (no fine-tuning) | 0.3460 / **−0.37** | 0.3245 | 0.2374 | 3/6 |
| Official run 1 (trained on Pa, discarded) | — | 84.6 | 94.5 | 2/6 |
| Official run 2 ep57 (non-dimensional) | 0.2298 / **+0.81** | — | — | — |
| **Current decoder-30ep** | **0.0929 / +0.91** | **0.0629** | — | **5/6** |

Official run 2 flipped rank correlation from −0.37 to +0.81 in 57 epochs. Absolute MAE
is still catching up, so it runs to 560 epochs and the **checkpoint with minimum
validation loss** gets selected.

**Gate integrity**: the 15 standard-gate cases have **no intersection** with the
official run's 61 training / 9 validation cases (verified).

---

### 2026-08-27 (2nd) — Official pipeline, third round: two defects fixed, rematch

Even after fixing the first round's units mismatch, the gate MAE stayed flat at 0.19.
Digging further turned up two more defects.

**Defect A — the shear columns needed dividing too.** Only the pressure was divided by
ρU², on the judgement that shear was "already non-dimensional." That judgement was
wrong. In the source data friction carries **0.01%** of the drag integral (I had
measured it and walked past it), so dividing pressure alone inflates the friction share
to about **10%** and biases every coefficient. The current model's code
(`finetune_domino_v3.py:72`) had been dividing **the whole array** from the start.
Effect of the fix (same epoch 11): **0.2312 → 0.1642**, rank +0.62 → +0.83.

**Defect B — the train/validation split was degenerate along the drag axis.**
Re-measuring with `audit_split_spread.py`: of the 54 training cases, **40 sit in Cd
0.23-0.30** with quartiles 0.248/0.248/0.248, and **the 9 validation cases span Cd
0.2772-0.2991 (std 0.0058)** — effectively a single point. That is why validation loss
went flat from epoch 31 while the gate score kept moving, and why the validation set
could support neither model selection nor a calibration fit.
`build_stratified_split.py` rebuilds it stratified by Cd: 65 training cases
(0.1999-0.6059), 7 validation (std 0.0460, 8× better). Effect (epoch 11):
**0.1642 → 0.1186**.

**Side finding — 10 cases with a different axis convention.** During stratification,
run_29/73/82/83/84/105/107/119/120/121 turned out to sit far away in world coordinates
(x≈10 z≈20, z≈90, etc.) with dimensions [4.06, 3.74, 4.99] — not vehicle shapes
(the normal group is [4.99, 1.96, 1.32]). The official training pipeline uses one
bounding box shared by all cases, so these contribute **zero points inside the box**.
The existing 70-case set is entirely clean, so earlier results are uncontaminated.
Translation alone does not fix them — the rotation convention would have to be
determined — so they were excluded this time, at the cost of losing
**run_107 (Cd 0.7875), the highest-drag case in the pool**. Left as a separate task.

**Freezing had no effect.** Adding `train.freeze_modules` to `train.py` and freezing
`geo_rep_surface` gave a best of 0.1163 against 0.1186 unfrozen — inside the noise
range (±0.03). Reasonably so, since the encoder is only **2.9%** of the parameters
(288K/10.06M). The hypothesis that "training the encoder as well destroys the
representation" is rejected.

**Final comparison** (each model's best checkpoint)

| Model | Gate15 MAE/rank | Windsor MAE | gtr Cd MAE | ΔCd direction | ΔCd MAE |
|---|---|---|---|---|---|
| pretrained (no fine-tuning) | 0.3460 / −0.37 | 0.3245 | 0.2374 | 3/6 | — |
| Official run 1 (trained on Pa) | ~94.5 | 84.6 | 94.5 | 2/6 | — |
| Official stratified ep11 | 0.1186 / +0.874 | 0.0800 | 0.0845 | 4/6 | 0.0119 |
| Official stratified + frozen ep5 | 0.1163 / +0.853 | 0.0851 | 0.0517 | 4/6 | 0.0083 |
| **Current decoder-30ep** | **0.0929 / +0.91** | **0.0629** | **0.0121** | 4/6 | **0.0068** |

**Verdict: no promotion.** The official pipeline went from "broken" to "competitive,"
but the current model leads on every gate. Both variants also peak at epochs 5-11 and
degrade thereafter (overfitting on 65 cases) — the official 560-epoch schedule does not
fit our scale.

**The most important finding — both models over-amplify the drag spread.**
Fitting pred = a·true + b on the 15 gate cases:

| Model | Raw MAE | Slope a | Intercept b | Residual after linear correction |
|---|---|---|---|---|
| Official frozen ep5 | 0.1443 | 2.188 | −0.417 | 0.0771 |
| **Current decoder30** | **0.0929** | **1.595** | **−0.240** | **0.0515** |

Low drag is pushed lower and high drag higher (run_78 true 0.5035 → current 0.7210;
run_52 true 0.1807 → 0.0624). A substantial part of the raw MAE gap is
**linear miscalibration, not model ability**. This applies to the model in production
too, so the magnitude of change reported when a control point moves may be exaggerated.
Fitting a calibration on the validation set was impossible because of defect B
(validation std 0.0058); now that a stratified split exists it is possible —
**top priority next**.

---

### 2026-08-27 (3rd) — Fixing the scoring criterion reversed the verdict

User's observation: **"Cd isn't the problem — it's whether moving a control point makes
things better or worse. That's it. Getting Cd exactly right is extremely hard; that
comes last."**

A fair point, and the verdict up to then ("no promotion") was **scored purely on
absolute Cd gates**. Re-scoring on the question the product has to answer reverses the
conclusion.

**ΔCd must be scored against LES.** This project's division of labour was always
**SU2 = absolute values / LES = ΔCd truth** (the `les-solver-for-g3-labels` note), yet
direction was still being scored against G2 labels. `benchmarks/gtr_smooth_pairs.json`
already contained the LES ΔCd, and 2 of those pairs have the opposite sign from G2
(roof_lower_10mm: G2 +0.0095 / LES −0.0262, sign reproduced across 3 configurations;
rear_wide_25mm: G2 +0.0088 / LES −0.0029).

| Model | Direction vs G2 | **Direction vs LES** | **Magnitude rank corr. with LES** |
|---|---|---|---|
| **Current decoder-30ep (in production)** | 4/6 | 4/6 (chance 34%) | **−0.14** |
| Official stratified ep11 | 4/6 | **6/6** (chance 1.6%) | **+0.60** |
| Official frozen ep5 | 4/6 | **6/6** (chance 1.6%) | **+0.60** |

Against G2 the three are indistinguishable; against LES the official-pipeline models
call all six deformations correctly. The roof_lower_10mm case all three "got wrong"
was **the G2 label under suspicion**, and the official models sided with LES.

**Magnitude rank matters more.** The current model is at **−0.14** — effectively no
signal — while the official models are at **+0.60**. Answering "which control point is
more worth moving" needs magnitude order, not just direction.

**Limits (stated honestly)**: 5 of the 6 LES deltas are negative, so **answering
"always improves" already scores 5/6**. The 6/6 is one better than that, and a rank
correlation of +0.60 at n=6 is not significant on its own. More pairs are needed to
settle it — though it is a different kind of number from the current model's −0.14.

**Side finding — the reference area is wrong by 3× on 3 gate labels.**
`audit_reference_area.py`: the badly missed cases declared a `ref_area`
**3.0-3.2× their actual frontal area** (run_52/58/79). Since Cd = force/(q·ref_area),
the prediction comes out small by exactly that factor. Undoing the ratio takes run_52
from **65% to 4%**. The gate's mean error goes 35% → 21% and within-±20% goes 6/15 →
9/15. This is a bookkeeping problem, not a model problem.

**Where absolute Cd stands (for reference, lowest priority)**: the current model's
median relative error is 24%, worst 74%, and **only 1 of 15 is within ±5%**. It cannot
be used as an absolute predictor. run_55/56/81 all come out at an identical 0.2345 —
it cannot even tell them apart.

**Next priorities (re-ordered)**
1. **More pairs** — LES-based scoring is confirmed valid, and 6 pairs cannot separate
   6/6 from 5/6 ("always decreasing"). The design criteria already exist
   (|ΔCd| ≥ 0.0015 watertight; G2 first → watertight surface → LES).
2. **Clean up the reference-area bookkeeping** — 3 errors of 3× in the gate alone means
   the whole training pool needs sweeping.
3. Absolute Cd calibration (the slope-1.6 over-amplification) — after the first two.

---

### 2026-08-27 (4th) — Going from 6 pairs to 83 revealed the 6/6 as chance

While starting the pair expansion by searching S3 and G3_TEST, **what was being looked
for was already there**: `~/g3-v2/data/g2-s3-deformation-pairs-v1` held **403 G2
deformation pairs** (118 of 138 jobs) stored as base/target point clouds, displacement,
base_cd, target_cd, delta_cd — and the full surface field of every RBF design step
(`surface_flow.vtu`, 741 KB each) was still in S3. **The 32-hour LES campaign was
unnecessary.**

Of the 89 pairs passing the filters (|ΔCd| ≥ 0.0015 noise floor, Cd 0.10-1.00 physical
range, at most 4 per geometry), the **83 pairs across 31 geometries** whose surface
fields exist in S3 became the benchmark
(`benchmarks/g2_deformation_pairs_83.json`, median |ΔCd| 0.00593).

**Result — all three models fall short of the majority-direction baseline.**

| Model | Direction | p vs majority (73%) | Rank corr | Value corr | ΔCd MAE |
|---|---|---|---|---|---|
| **Current decoder-30ep (in production)** | 48/83 = **58%** | 0.9993 | +0.10 | +0.05 | 0.0156 |
| Official stratified | 48/83 = **58%** | 0.9993 | **+0.35** | −0.01 | 0.0147 |
| Official frozen | 56/83 = **67%** | 0.9121 | +0.13 | −0.27 | 0.0163 |

Answering "always improves" scores 61/83 = **73%**. The best model manages 67%.

**This corrects the verdict in section (3).** The 6/6 from the 6 LES pairs was a
small-sample artefact, and the limit recorded at the time ("5 of 6 are decreasing, so
always-decreasing also scores 5/6; +0.60 at n=6 is not significant on its own") played
out exactly. It cannot support a promotion decision.

**The failure is closer to weak response than to being wrong.**

| Model | Median predicted \|ΔCd\| | vs truth | Almost no response (<0.001) |
|---|---|---|---|
| Current | 0.00224 | 0.38× | **31/83** |
| Official stratified | 0.00449 | 0.76× | 13/83 |
| Official frozen | 0.00374 | 0.63× | 16/83 |

Magnitudes are roughly right at 0.4-0.8× the truth, but the sign is near random. In
particular, **the model in production barely changes its prediction on 31 of 83 pairs
when the control points move.** Not because the deformations are subtle — in these
pairs a median 2,102 of 7,618 points move, with a median maximum displacement of 0.034 m.

**On the label-reliability objection**: these 83 pairs carry G2 labels, and G2 has a
record of disagreeing with LES on 2 of 6. But with label error rate e, a perfect model
scores (1−e) while the majority baseline is computed from the observed distribution, so
**beating 73% remains the correct bar**. Label noise pushes everyone toward 50% without
invalidating the baseline.

**Next**: (a) why the sign is wrong — apply `evaluate_surface_delta.py`, which looks
directly at the local pressure response in the deformed patch, to these 83 pairs
(b) put these 83 pairs into **training** (pair training has so far used only 6)
(c) spend LES only on pairs where the model and G2 disagree, making the 4-hour runs
count.

---

### 2026-08-27 (5th) — Training on 83 pairs: the ΔCp loss works, the ΔCd loss I added hurt

**A verification that came out of data preparation**: the per-design-step `history.csv`
is the **adjoint history** (Sens_Geo, rms[A_P]), not the primal, so it has no CD column.
Drag was filled in from the pair manifest and lift computed by integrating the surface
field (`backfill_pair_lift.py`). Integrating drag the same way **matches the manifest to
a median 0.7%** (ratio 1.007, quartiles 0.998/1.008), which validates the whole
download → conversion → field path.

**Split by geometry** (`split_pairs_by_geometry.py`). The pairs within a job are
successive optimisation steps of the same car, so splitting by pair would measure
memorisation. 18 geometries / 52 pairs to train, 11 geometries / 31 pairs to test,
**zero run overlap**, with majority direction 73%/74% on each side.

**Loss design hypothesis and result**: today's diagnosis was "magnitude right at
0.4-0.8×, sign random," so a **paired ΔCd loss** penalising the integrated difference
the product reports was added (`--delta-cd-weight`, scaled so an error of 0.01 costs
unity). **The hypothesis was rejected.**

Held-out 11 geometries / 31 pairs (majority 23/31 = 74%):

| Model | Direction | p vs majority | Rank corr | Value corr | ΔCd MAE |
|---|---|---|---|---|---|
| Current decoder-30ep | 18/31 = 58% | 0.985 | +0.25 | +0.25 | 0.0112 |
| Official stratified | 17/31 = 55% | 0.994 | +0.57 | +0.47 | 0.0099 |
| **dCp only (52-pair training)** | **21/31 = 68%** | 0.848 | +0.49 | **+0.79** | **0.0071** |
| dCp + dCd | 17/31 = 55% | 0.994 | +0.19 | +0.18 | 0.0081 |

**52-pair ΔCp training clearly works**: value correlation +0.25 → **+0.79**,
ΔCd MAE 0.0112 → **0.0071 (37% better)**, direction 58% → 68%. Raising pair training
from 6 pairs to 52 finally gave it material to work with. Adding the ΔCd loss on top
degrades every metric — the campaign-one lesson from section (3), that putting the
integral term in directly causes overfitting, repeats in the pair loss.

**Key finding — what was hiding the sign was a systematic bias in the predicted ΔCd.**
The dCp-only model **consistently over-predicts ΔCd by +0.015**. With a median |ΔCd| of
0.006, that flips the sign of every pair near zero. Applying a mean bias fitted on the
training pairs to the held-out set (`test_delta_calibration.py`):

| Correction | Direction | p vs majority |
|---|---|---|
| None | 21/31 = 68% | 0.848 |
| **Mean bias removed** | **26/31 = 84%** | **0.152** |
| Linear correction | 23/31 = 74% | 0.594 |

**The majority baseline is beaten for the first time.** Most of the gain is explained by
the smallest-|ΔCd| band (13 pairs) going from 7/13 to 10/13. The same prescription helps
the official stratified model (55% → 84%) and hurts the current model (58% → 48%) and
dCp+dCd (55% → 35%) — a property of particular models, not a universal artefact.

**Limits and verification procedure**: 4 models × 3 corrections = 12 combinations were
examined on 31 pairs, so the 84% (p=0.15) could be chance. **Three-fold cross-validation
by geometry** is re-measuring with every one of the 83 pairs held out exactly once
(`crossval_pairs.py`).

**Cross-validation result — the bias correction is confirmed.** Re-measured with three
geometry folds putting all 83 pairs on the held-out side exactly once:

| Correction | Direction | p vs majority (73%) | Value corr | ΔCd MAE |
|---|---|---|---|---|
| None | 54/83 = 65% | 0.966 | +0.24 | 0.0130 |
| **Mean bias removed** | **72/83 = 87%** | **0.0028** | +0.19 | 0.0165 |

**This is the first statistically significant result on the product question ("does
moving a control point help?").** The fitted bias per fold is consistently positive at
+0.00595 / +0.01353 / +0.01238.

**Three caveats to attach honestly**:
1. The bias correction **buys direction at the cost of magnitude** — ΔCd MAE 0.0130 →
   0.0165. Shifting every prediction by a constant fixes signs near zero but moves the
   magnitudes further away. It does nothing for "which control point helps most"
   (magnitude ranking).
2. The **value correlation of +0.79 from the single split was that fold's luck** —
   across all 83 pairs it is +0.24. Magnitude tracking is still weak.
3. The bias constant varies more than twofold across folds, +0.006 to +0.014. That means
   **its value depends on which data it is fitted to**, so its stability across shape
   families must be checked before it goes anywhere near the service.

**In summary**: direction is usable at 87% (always-improves = 73%), magnitude ranking is
not. Next: (a) find the cause of the bias — G2 re-meshes each variant, so the predicted
ΔCd may carry a mesh-change component rather than a shape-change one (b) train for
magnitude ranking (c) measure the bias constant's stability per shape family.

---

### 2026-08-27 (6th) — Root cause: the geometry encoder cannot see the deformation (the grid is coarser than the deformation)

**First, whether the 87% is safe.** Because 73% of this benchmark is "decreasing," a
model predicting ΔCd near zero would make `mean(pred − true)` automatically positive,
which would reduce "bias removal" to injecting a prior. `audit_bias_origin.py`:

| | Pairs called "increase" | Correct among them | Precision |
|---|---|---|---|
| No correction | 49 | 21 | 43% |
| **Bias removed** | **13** | **12** | **92%** |
| Always decreasing | 0 | — | — |

Actual increases are 22/83 (chance 27%), and after correction the model calls only 13
increases and gets 12 right. **This is discriminating signal, not prior injection** —
the 87% holds.

**The bias had two components.** Per fold, `bias = mean(pred) − mean(true)`, and
mean(pred) is **consistently positive** at +0.0038/+0.0047/+0.0054. One component is the
benchmark's skew (−mean(true), which varies by fold); the other is a model-intrinsic
offset of +0.0045.

**The mesh hypothesis is rejected.** `test_mesh_hypothesis.py`: in these pairs
**the cell count does not change at all and the reference area is identical** — RBF moves
points rather than re-meshing, so the topology is preserved. (The gtr dataset does
re-mesh, and that experience was wrongly generalised.) The only thing that changes is
wetted area (mean −0.1%); when wetted area falls, the true Cd falls (−0.24) but the
residual grows (+0.22) — meaning **the model under-responds to shape change**.

**The real cause — the encoder cannot see the deformation.** `probe_geometry_encoding.py`:

| Relative encoding distance | Median |
|---|---|
| Within a pair (same car, before/after deformation) | **0.00146** |
| Between different cars | 0.82444 |
| **Ratio** | **0.002** |

A deformation moves the encoding by **0.2%** of what separates different cars. Some pairs
are at exactly zero. The correlation between encoding distance and |true ΔCd| is −0.19,
i.e. none. **The signal the decoder should read never arrives.** Whatever change the
prediction shows comes from the per-cell inputs (centres, normals, areas), not the global
encoding.

**Mechanism — the grid cell is larger than the deformation.** `measure_grid_waste.py`,
`interp_res` [128, 64, 64], median maximum deformation **4.02 cm** (quartiles 2.11-5.18):

| Box | Cell size (cm) | Deformation/cell | Verdict |
|---|---|---|---|
| DrivAerML default | [5.08, 4.38, 2.69] | [0.79, 0.92, 1.5] | not visible |
| **Our training config** | [6.17, 4.69, **9.06**] | [0.65, 0.86, **0.44**] | not visible |
| Box fitted to the car | [4.49, 3.35, 8.81] | [0.90, 1.2, 0.46] | not visible |

**In every configuration the deformation is smaller than one grid cell.** Compounding it,
the shapes' z range is −0.58 to 4.79 m while a car is 1.47 m tall — the cases sit at
different world positions, so the union box is **3.7× inflated** and that much z
resolution is thrown away.

**This does not contradict "freezing had no effect" in section (2).** That experiment
measured absolute Cd across *different cars*, where the encoder works well (between-shape
distance 0.82). It is blind to deformations of *the same car* — a different metric gives
a different conclusion.

**Prescriptions (in order of cost-effectiveness)**
1. **Per-case frame normalization** — free. The z span goes 5.37 m → 1.47 m, recovering
   **3.7×** of resolution. The current box's 9.06 cm z cell becomes 2.4 cm.
2. **Raise `interp_res`** — 256×128×128 halves the cell on every axis. It costs 8× in
   the geometry representation only, a small share of the whole model.
3. Both together: cells of about [2.1, 1.6, 1.15] cm → the deformation becomes 1.9/2.5/3.5
   cells, resolvable for the first time.
4. If still insufficient, feed the displacement field explicitly as an input channel.

---

### 2026-08-27 (7th) — Prescription 1 applied: it was not resolution, **the shapes were outside the grid**

Measuring the normalized coordinates directly with `check_normalized_placement.py` showed
something far worse than section (6)'s diagnosis. The encoder normalizes against a fixed
DrivAerML box and then samples the SDF over a [−1, 1] grid — and our shapes had a
normalized z of **+3.41 to +4.94**.

| | Vertices inside the normalized box [−1,1] |
|---|---|
| Before alignment | **12.6%** (most cases **0.0%**) |
| After alignment | **100.0%** |

**With coordinates outside the grid, any shape produced essentially the same encoding.**
That is the real reason a deformation moved the encoding by only 0.2%. Per-case shifts
also came out spread across x −10.45 to +1.91 m and z −18.17 to +2.18 m — the data mixes
several coordinate conventions.

**Alignment method** (`align_case_frames.py`): **translation only** — no rotation, no
scaling — so forces, areas, and coefficients are unchanged. The anchor is deliberately
insensitive to deformation: the **area-weighted centroid** (x, y) and the **ground plane**
(z). Using a bounding-box extremity would let a 4 cm tail deformation shift the box by
4 cm and erase the very signal being measured.

**Effect**

| | Before | After |
|---|---|---|
| Encoder sensitivity (within-pair / between-shape) | 0.002 | **0.021 (14.6×)** |
| **Pretrained absolute Cd MAE** | 0.477 | **0.331 (31% better)** |
| Fine-tuned absolute Cd MAE | 0.085 | 0.085 (unchanged) |
| Direction, no correction | 54/83 = 65% | **57/83 = 69%** |
| Direction, bias removed | 72/83 = 87% (p=0.003) | 67/83 = 81% (p=0.083) |

**The pretrained model improving 31% is clear evidence that alignment fixed a real
defect.** The absence of change after fine-tuning means fine-tuning had already been
compensating for the frame error, and the capacity spent on that compensation is now
freed. But **direction accuracy still falls short of the 73% majority baseline** —
prescription 1 alone is not enough.

**Prescription 2 (raising resolution) is worthless — rejected by measurement.** The
pretrained weights load at `interp_res` 256×128×128 with **0 missing and 0 shape
mismatches**, so the experiment was free, and encoder sensitivity barely moved: 0.021 →
**0.024**.

In hindsight that number is reasonable. A 4 cm deformation is about a 1% shape change on
a 5 m car, and the actual ΔCd is 0.006/0.3 = 2%. A 2% move in the encoding is
proportionate. **After alignment the encoder is not the bottleneck.**

**Remaining bottleneck and next move**: the decoder has to turn a 2% encoding change into
the right *sign*, and most of its training is absolute-Cd data where that distinction did
not matter. The most direct lever is **more training pairs** — of the 400 pairs, the 83
used as a benchmark are what remains after the noise floor (|ΔCd| ≥ 0.0015) cut 276, and
**that floor is needed for the benchmark but not for training** (unbiased noise is still
a valid regression target). Training on all ~370 physically valid pairs while scoring on
the clean 83 gives 4.5× the material.

---

### 2026-08-28 — Five times the training pairs: direction unchanged, **the 87% is retracted**

The training pairs were expanded as section (7) prescribed. The noise floor applies to
scoring only, not training — the winning method is the **ΔCp field loss**, which never
reads the ΔCd value. Fetching 437 more meshes from S3 gave **384 runs / 298 pairs / 85
geometries**, with scoring kept on the clean 83 pairs (29 geometries). Each fold trains
on everything except the scoring geometries, with run overlap checked at zero
(`crossval_big.py`).

Preparation checks: on the new 384 runs the **integrated Cd matches the manifest within
0.5%**, and after alignment **100% of vertices are inside the encoder grid**. Coefficients
were recomputed in the aligned frame — the normal-orientation test depends on the
coordinate origin, so a shape far from the origin can be flipped incorrectly.

| Condition | Training pairs | No correction | Bias removed | Fitted bias |
|---|---|---|---|---|
| Before alignment | 52 | 65% | **87%** (p=0.003) | +0.006 to +0.014 |
| After alignment | 52 | 69% | 81% (p=0.083) | +0.007 to +0.015 |
| **Aligned + expanded** | **262-267** | **69%** | **71%** (p=0.737) | **+0.0013 to +0.0023** |
| Majority baseline | | **73%** | | |

**Five times the training pairs leaves direction accuracy unchanged at 69%.**

**The 87% is retracted.** That the bias-corrected result swings 87% → 81% → 71% across
conditions is itself the evidence that it cannot be quoted as a capability. The precision
on "increase" calls collapsed the same way, from 12 of 13 (92%) at 52 pairs to 10 of 22
(45%) at 298 — that 92% was small-sample too. Recording it in sections (5) and (6) as
"the first significant result on the product question" was overreach.

**But the expansion did settle one thing**: the fitted bias shrank sixfold
(+0.007-0.015 → +0.0013-0.0023). **It was a small-sample artefact, and more data removed
it on its own.** And with the bias gone, direction is still 69% — so the gain the bias
correction had been providing was not the model's own signal.

**Current state**: the uncorrected signal is stable at 65%/69%/69% across three
conditions and **falls short of the 73% majority baseline**. What has been tried and
established:
- Loss design (ΔCp / ΔCp+ΔCd / integrated Cd): ΔCp alone is best; the integral term hurts
- Training pairs 6 → 52 → 298: improvement up to 52, flat beyond
- Frame alignment: **fixed a real defect** (pretrained absolute Cd 31% better, encoder
  sensitivity 14.6×) but direction only 65% → 69%
- Grid resolution ×2: sensitivity 0.021 → 0.024, meaningless
- Bias correction: unstable, retracted

**What remains, and the judgement**: after touching loss, data volume, coordinate frame,
and resolution, it is still 69%. This may well be a reliability problem in the G2 ΔCd
labels themselves — G2 disagreed with LES on 2 of the 6 gtr pairs (20-33%), and label
error of that order is consistent with the observed 69%.
**The next move is not more model work but validating the labels**: taking the pairs
where the model and G2 disagree and adjudicating with 4-level LES separates a model limit
from a label limit. It is the one use that makes a 4-hour run worth it.

---

### 2026-08-28 (2nd) — The 69% was a mixture: **76% on actual cars** (established without LES)

While selecting pairs for LES adjudication, **6 of the 13 disagreements turned out to be
concentrated in just two geometries** (b2K5nBxC 4, 8pzSt7MP 2). Not random. And checking
dimensions while preparing to seal the shapes:

- Control 5ikuXj53: [5.00, 1.97, 1.32] m, volume 7.85 m³ — a vehicle
- **b2K5nBxC (4 disagreeing pairs)**: **[4.06, 3.74, 5.00] m, volume 35.9 m³** — not a vehicle

**A hypothesis testable for free before spending 32 GPU-hours**: the 69% ceiling may be a
**shape distribution** problem rather than a label problem.

**Splitting the score with `aox_g3/upload_gate.py`** (a production classifier written for
another purpose before this analysis) — the point being that this is not a threshold
chosen after seeing the result:

| Class | Pairs | True direction | Majority | Model | p | Value corr |
|---|---|---|---|---|---|---|
| **full_car** | 63 | 41 down / 22 up | 65% | **48/63 = 76%** | **0.040** | **+0.62** |
| non_car_shape | 16 | 16 down / **0 up** | 100% | 5/16 = 31% | — | −0.33 |
| component | 4 | 4 down / **0 up** | 100% | 4/4 = 100% | — | +1.00 |

**The overall 69% and the 73% majority baseline were a mixture produced by the degenerate
non-car group.** The 16 non_car_shape and 4 component pairs all move **one way (down)**,
so they carry no direction information — they cannot separate a good model from a bad one,
and saying "increase" is automatically penalised. Those 20 pairs pushed the majority
baseline up to 73% while pulling the model's score down.

**On the 63 actual cars: 76% against a 65% majority, p = 0.040, value correlation +0.62.**
Compared with +0.23 across all 83 pairs, it is clear where the signal was.

**A stricter shape filter** (height 1.2-1.8 m, width 1.7-2.3 m) gives, on 45 pairs,
**89% against a 51% majority (23 down / 22 up, nearly balanced), p ≈ 0, and 18 of 19
"increase" calls correct (95% precision)**. But that threshold was chosen after seeing the
result, so **the number quoted is the independent classifier's 76%.** Both criteria lead
to the same conclusion.

**The LES adjudication is cancelled.** The top adjudication candidate, b2K5nBxC, is
exactly `non_car_shape` (3.85 m wide, 3.70 m tall) — labels were about to be adjudicated
for something that is not a car. **A shape-class check saved 32 GPU-hours.**

**Correction summary**: the 87% from sections (5) and (6) stays retracted, and the "69%
ceiling" of sections (7) and (8) is updated to **"69% mixed across shapes, 76% on actual
cars."** The hypothesis that label reliability is the ceiling remains untested, and the
current evidence favours **shape distribution**.

**Next priorities**
1. **Separate non-car shapes out of the benchmark** — scoring with 20 direction-free pairs
   mixed in understates the model. Freeze the `upload_gate` verdict into the benchmark
   metadata.
2. **A policy for non-car shapes** — direction prediction on such shapes is at chance
   (31%), so expose the verdict the `upload_gate` already produces in the inference
   response as a confidence warning (that path already exists in `/v1/infer`).
3. Retrain and re-score on the 63 car pairs alone to see whether 76% goes higher.

---

### 2026-08-28 (3rd) — Training on car geometry only: 45% better magnitude accuracy, direction unchanged

Classifying every run with `aox_g3/upload_gate.py` (`classify_pair_runs.py`) gives
**full_car 313 / non_car_shape 58 / component 6 / unsure 6** of 383 runs, and
**244 full_car of 298 pairs** — only 18% data loss. A shape filter was added to the
cross-validator so that training and scoring both use cars only.

| Condition | Training pairs | Direction | p | Value corr | **ΔCd MAE** | Fitted bias |
|---|---|---|---|---|---|---|
| All shapes → car scoring | 262 | 48/63 = 76% | 0.040 | +0.62 | 0.0110 | +0.0013 to +0.0023 |
| **Cars only → car scoring** | 218 | 49/63 = **78%** | **0.021** | **+0.68** | **0.0060** | **−0.0004 to −0.0007** |

**Direction moved by one pair, which is not read as an improvement** — applying the
criterion pinned down before the run ("a few pairs' difference out of 63 is noise"). The
real gains are two: **ΔCd MAE fell 45%** (0.0110 → 0.0060, the same size as the median
|ΔCd| of 0.006), and **the bias vanished entirely** (−0.0005). That bias removal now makes
the score worse (78% → 75%) is corroboration that the bias really is zero.

**Re-measuring the current model — correcting an earlier report.** On the same 63 car
pairs the current decoder-30epoch scores **52/63 = 83%**, value correlation **+0.82**,
ΔCd MAE 0.0055. Section (4)'s statement that "the current model is at 58% with value
correlation −0.14, i.e. no signal" was **a figure diluted by direction-free non-car pairs**.
The model in production does well on actual cars.

**That 83% cannot be quoted as generalization, though.** Digging into the current model's
training log shows it read 71 reaudit runs, and **8 of their 43 S3 job UIDs overlap with
the car benchmark's 22 geometries**. Keeping only unseen geometry:

| Model | All 63 pairs | 41 pairs unseen by the current model (majority **93%**) |
|---|---|---|
| Current | 83% (baseline 65%, p=0.002) | 73%, value corr +0.54, MAE 0.0041 |
| Cross-validated model | 78% (baseline 65%, p=0.021) | 66%, value corr +0.34, MAE 0.0064 |

**The pairs carrying directional variety happen to be concentrated in the 8 geometries the
current model saw**, so removing them leaves 38 of 41 pairs moving one way (93% majority)
and the discriminating power disappears. Whether the current model's 83% is generalization
or recall **cannot be separated with this benchmark**.

**So what can be quoted**: the cross-validated model's **78%** (fully geometry-separated,
65% majority, p=0.021, value correlation +0.68, ΔCd MAE 0.0060). The current model's 83%
stays as a reference figure, always annotated with the training overlap.

**A structural limit of the benchmark surfaced**: of the 22 scorable car geometries, 8
overlap with the current model's training, and the remaining 14 are almost entirely
one-directional. **What the next benchmark expansion requires is not pair count but
"drag-increasing deformations on geometry the current model has not seen."** Either
re-select from the 400-pair pool under that condition, or, failing that, this is where LES
earns its place.

---

### 2026-08-28 (4th) — Re-selection: the pool is exhausted, and **78% on a balanced benchmark (p=0.0003)**

The 400-pair pool was re-selected under the requirement left by section (3) —
"drag-increasing deformations on geometry the current model has not seen"
(`find_drag_increasing_pairs.py`).

| Stage | Pairs remaining |
|---|---|
| All | 400 |
| Physically valid | 339 |
| **Drag increasing** | **89** |
| Above the noise floor | 24 |
| **+ unseen by the current model** | **3** (2 geometries) |

**The pool is exhausted.** Only 3 pairs, one of which barely clears the floor at +0.0019.
The reason is structural — the 400 pairs are products of RBF *optimisation* runs, so by
construction most of them reduce drag.
**Cleanly validating the current model with the existing data is impossible.**

**But a better benchmark for comparing candidate models could be built.** Dropping the
"unseen by the current model" condition leaves 24 increasing pairs, and matching them with
an equal number of decreasing ones, `build_balanced_benchmark.py` produced
**40 pairs / 17 geometries / 20 up : 20 down, a 50% majority baseline**
(`benchmarks/g2_balanced_pairs_40.json`).
In the previous 63-pair set (65% baseline) only 22 pairs contributed discriminating power;
here all 40 do. Cross-validation was re-run with this as the scoring set (some of the 17
geometries were in the previous training folds, so a re-run was necessary).

| | Direction | p vs majority (50%) | Value corr | "Increase" precision | Fitted bias |
|---|---|---|---|---|---|
| **No correction** | **31/40 = 78%** | **0.0003** | +0.52 | 18 of 25 (72%) | −0.0009 to −0.00006 |
| Bias removed | 33/40 = 82% | ~0 | +0.52 | 20 of 27 | — |

**This is the cleanest result of the session**: a balanced benchmark (every pair
discriminates), full geometry separation, **78% with no bias correction**, and a fitted
bias of effectively zero. That the 78% matches what the 63-pair car benchmark gave also
supports its stability. Bias correction is no longer needed — what was introduced as a
stopgap in section (5) became unnecessary once the data was adequate.

**Where LES is needed became precise for the first time.** The twice-cancelled campaign is
now not "adjudicating existing labels" but **"creating and labelling drag-increasing
deformations on car geometry the current model has not seen."** The physical prescription
is clear too — blunt the tail, widen the rear, raise the rear roofline. That is for
validating the current model; candidate-model development and comparison are fully served
by the balanced benchmark just built.

---

### 2026-08-28 (5th) — Sweep test: one meaningful deformation works, **a fine sweep does not**

The data for "change it a little at a time and test whether it is right" already existed —
the RBF design steps DSN_001 → 002 → 003 → … are a trajectory of progressive deformations
of one car, and the cross-validated predictions come from models that never saw that car
(`sweep_trajectories.py`).

**First, whether the test itself is valid.** The median |ΔCd| between consecutive steps is
**0.00011** — a **sixty-fifth** of the balanced benchmark's pairs (0.00713) — and **80%
falls below the G2 noise floor (0.0015)**. Worse, **the median total span of a trajectory
(max − min Cd) is 0.00068, and 68% of the 65 trajectories are narrower than the noise
floor** — for those, the labels cannot say which design is best, so there is nothing to
grade the model against.

Re-measured on only the trajectories the labels can resolve (span ≥ 0.0045, step ≥ 0.0015):

| Metric | Model | Baseline | Verdict |
|---|---|---|---|
| Optimal design exactly matched | 5/14 = 36% | chance 24% | marginal (n=14) |
| Optimal design within ±1 step | 6/14 = 43% | chance 47% | below baseline |
| **Median trajectory rank correlation** | **+0.40** | 0 | real signal |
| Step direction (significant steps) | 26/34 = 76% | majority 79% | at baseline |

**Product conclusion — use it for the right purpose.**
1. **One meaningful deformation** (|ΔCd| ≥ 0.0015, balanced up/down): **78%** (chance 50%,
   p = 0.0003). Usable.
2. **Finding the optimum in a fine sweep**: exact 36% / within ±1 43% (chance 47%).
   **Not usable.**
3. **The order of the response curve**: rank correlation +0.40. Partially tracked.

**Side finding — a problem with the G2 optimisation loop itself.** **68% of the existing
RBF optimisation trajectories have a total span narrower than the solver's own noise.**
Those optimisations ran without being able to resolve a real improvement. That is a matter
to check on the label-production side, before our model.

---

### 2026-08-28 (6th) — Frontend: measured confidence and error bars put on screen

Following the observation that everything was being measured with nothing reaching the
service, a check confirmed that **none of this session's model work was deployed** (the
serving checkpoint is still `decoder-30epoch.pt`). That is correct in itself, since there
was no clean win worth promoting. Instead, the work was **honestly moving what has already
been measured onto the screen**.

**Putting frame alignment into the inference path is deferred.** A/B result (current model):

| Gate | Original | Aligned |
|---|---|---|
| gate15 | 0.0929 | **0.0710** |
| Windsor | **0.0629** | 0.0886 |
| gtr | **0.0121** | 0.0209 |

The pretrained model improves greatly on 2 of 3 with alignment (Windsor 0.3245 → 0.1281)
while the current model improves on only 1 of 3. **That means the current model has been
compensating for the misaligned frames through fine-tuning**, so the gain from alignment
has to be taken by replacing the model wholesale after retraining.
The measurement itself was verified four ways — no normal-sign flips (0/7), bit-identical
field values, zero-variance point displacement (pure translation), and encoder and decoder
inputs moving together from 76.2% to 100%.

**ΔCl was measured for the first time** (`measure_delta_cl.py`). The balanced benchmark
already carried lift labels (integrated from the surface field, the same integration that
reproduces the manifest's drag within 0.7%).

| | Direction | Majority | p | MAE | Value corr |
|---|---|---|---|---|---|
| ΔCd | 78% | 50% | 0.0003 | 0.008 | +0.52 |
| ΔCl | 72% | 55% | **0.018** | 0.019 | **−0.06** |

**ΔCl beats chance on direction significantly, but its magnitude is uncorrelated with the
truth.** Only the sign should be read.

**What went into the frontend** (aox-next `origin/debug`, `cdac4d7d`/`c5ced775`/`0b326e62`)
- `RecommendationTrustNote`: displays the **actually measured** direction accuracy for that
  shape class per upload-gate verdict (cars 78%/72%, non-car at chance 31%, components and
  unsure unmeasured)
- ΔCd/ΔCl shown to three decimals instead of four, with **±0.008 / ±0.019** alongside
- **The "meaningful" threshold raised from 0.001 to 0.008.** The old 0.001 was a sixth of
  the measured error, so changes whose sign could flip were being called improvements —
  the most substantive part of this change
- Changes inside the error band are grey with `Too small to call` rather than green/orange.
  Colour no longer asserts a direction
- Subtitle "ranked by predicted Cd" → "sorted by predicted ΔCd". State the sort key without
  implying that the order is validated (fine-sweep optimum hit rate 36% vs chance 24%)

**No magnitude-dependent threshold was set** — per-band direction accuracy across 40 pairs
gives 4-21 pairs per band and was not monotone (100/67/67/100%), so it cannot serve as
evidence.

---

### 2026-08-31 — Retraining on aligned data: **absolute Cd 43% better, but promotion withheld**

Executing the next move left in section (6): "the gain from alignment has to be taken by
replacing the model wholesale after retraining." The promotion criteria were registered
before starting — ① balanced-40 direction ≥ 78% ② no regression on the three standard
gates ③ absolute Cd improved.

**Alignment effect (all 265 reaudit runs, `audit_alignment_coverage.py`)**

| | Before | After |
|---|---|---|
| Vertices inside the grid (mean) | 45.2% | **87.3%** |
| Runs completely outside (0%) | **96** | **0** |
| Runs below half | 120 | 48 |

**A third of the training data was entirely outside the encoder grid.** The remaining 48
are non-car shapes with dimensions like [4.06, 3.74, 5.00] that translation cannot bring
inside — a `--min-grid-coverage` filter was added to the trainer to exclude them.

**The A/B is clean.** The 71 runs actually used (train 54 / val 9 / test 8) were
reconstructed from the current model's training log, and all 71 clear 50% coverage after
alignment, so the filter removes nothing. Same split, same 30 epochs, same trainer —
**the only difference is frame alignment.**

**Standard gate results** (gate15 confirmed to have zero intersection with train/validation)

| Model | gate15 MAE/rank | Windsor | gtr Cd | ΔCd direction |
|---|---|---|---|---|
| Current decoder-30ep | 0.0929 / +0.91 | 0.0629 | **0.0121** | 4/6 |
| Retrained, original frames | 0.0954 / +0.84 | 0.0891 | 0.0063 | 4/6 |
| **Retrained, aligned** | **0.0529 / +0.83** | **0.0582** | 0.0183 | 3/6 |

**Aligned retraining improves gate15 by 43%** (0.0929 → 0.0529) and leads on Windsor too.
That the original-frame retrain lands near the current model (0.0954 vs 0.0929) is exactly
what a control should do, attributing the gain entirely to alignment.

**But promotion is withheld — criteria ① and ② were not met.**
- ② no regression: **gtr Cd regressed 51%, 0.0121 → 0.0183**, and ΔCd direction 4/6 → 3/6
- ① balanced-40 direction: 82% → **75%**, rank correlation +0.91 → **+0.57**

**The balanced 40 pairs showed their limit too.** Those 40 include geometry all three
models trained on. Removing the contamination leaves **only 23 pairs with the majority
degenerating to 87%** (3 up / 20 down), with no discriminating power. In other words,
**there is currently no instrument that can adjudicate a promotion on the product metric** —
the pool exhaustion identified in section (4) shows up here as a real cost.

**Verdict**: frame alignment is a **real and large gain for absolute Cd** (gate15 43%) but
**a loss for ΔCd direction and ranking**. The two metrics move in opposite directions, so a
single model cannot take both, and since the product needs direction, **keeping the current
model is correct**.

**Two next moves**
1. **Combine alignment with pair training** — this retrain used only the absolute Cd loss.
   It is worth checking whether adding the ΔCp pair loss on the aligned data can make up
   the direction loss.
2. **Secure an instrument** — drag-increasing deformations are needed on car geometry the
   current model has not seen. The existing pool holds three, so **creating them with LES
   is the only path**.

**Combining alignment with pair training (continued the same day)**

Next move ① from the previous section was executed. To avoid increasing gate contamination,
**only the 42 car pairs corresponding to the 27 geometries the current model already
trained on** were selected — no new shapes at all, only a new signal (the ΔCp field
difference) on the same shapes. The pair runs go into training only, with validation and
test kept identical to the current model's (`merge_pair_into_train.py`,
`build_combined_split.py`). Training is 109 runs = 54 absolute + 55 pair.

| Model | gate15 MAE/rank | Windsor | gtr Cd | gtr ΔCd direction |
|---|---|---|---|---|
| **Current decoder-30ep** | 0.0929 / +0.91 | **0.0629** | **0.0121** | 4/6 |
| Retrained, aligned | **0.0529** / +0.83 | 0.0582 | 0.0183 | 3/6 |
| Combined dw1 | 0.0585 / +0.80 | 0.0783 | 0.0391 | 4/6 |
| Combined dw3 | 0.0698 / +0.89 | 0.0979 | 0.0557 | 4/6 |

**Combining recovers gtr ΔCd direction from 3/6 to 4/6 but badly degrades gtr Cd**
(0.0183 → 0.0391). It also gives back part of the gate15 gain alignment had on its own
(0.0529 → 0.0585). dw3 is worse — the larger the pair loss, the more the absolute accuracy
collapses.

**The 98% on the balanced 40 is contamination.** Combined dw1 scored 39/40 = 98%, but on
inspection **17 of the 42 pairs put into training are the same pairs the benchmark scores
on**. At the run level 23 overlap. Keeping only the 23 non-overlapping pairs degenerates
the majority to 87% (3 up / 20 down) with no discriminating power — for reference, combined
dw1 scores 22/23 there, but at p = 0.179 against the majority, not significant.

**Verdict: no promotion, keep the current model.** No candidate cleared registered
criterion ② (no regression). Three things are established:
1. **Frame alignment is a real, large gain for absolute Cd** (gate15 43% better).
2. **That gain conflicts with ΔCd direction.** Recovering direction with the pair loss
   collapses absolute accuracy by the same amount. No way to have both in one model has
   been found yet.
3. **There is no instrument to judge with.** The balanced benchmark overlaps with training
   geometry, and removing the overlap makes it degenerate. This is now the largest
   bottleneck — more models can be built, but nothing can say whether they are better.

**So the next step is the benchmark, not the model.** Drag-**increasing** deformations are
needed on car geometry the current model has not seen, and the existing 400-pair pool holds
three, so it is exhausted. Creating them with LES is the only path, and that campaign is no
longer a nice-to-have — it is **the single thing blocking further progress**.

---

### 2026-09-01 — LES benchmark campaign: 32 GPU-hours, one drag-increasing pair

The campaign section (4) identified as "the single thing blocking further progress"
was run: create and label drag-increasing deformations on car geometry the serving
model has never seen. Two cars × (baseline + 3 variants) = 8 runs of 4-level LES,
split across G3_TEST (carA) and G4_TEST2 (carB), about 4 hours each.

**Shape selection** (`select_les_base_shapes.py`, `dedupe_base_shapes.py`): 44
candidates that are full_car, watertight, and outside the serving model's training
set. Hashing the geometry showed those 44 collapse to **10 distinct shapes and only
5 distinct car types** — the pool's diversity is far smaller than its case count
suggests. carA (Cd 0.2479) and carB (0.4518) were taken from opposite ends.

**Deformations** (`make_drag_up_variants.py`): three smooth compact bumps designed
to attack the wake — blunt_tail (square off the rear), wide_rear (flare the rear
quarters), raise_roof (lift the rear roofline), 29-55 mm displacement, all verified
watertight and still classified full_car.

**Results**

| Car | Variant | ΔCd | S/N | Frontal area | Drag force | Verdict |
|---|---|---|---|---|---|---|
| carA | **blunt_tail** | **+0.0090** | **2.4** | +0.0% | **+3.3%** | **increase ✓** |
| carA | wide_rear | −0.0068 | 1.6 | +4.5% | +1.9% | inconclusive |
| carA | raise_roof | −0.0053 | 1.5 | +0.1% | −1.9% | inconclusive |
| carB | blunt_tail | −0.0035 | 0.4 | +0.0% | −0.5% | inconclusive |
| carB | wide_rear | −0.0358 | 3.7 | +1.5% | −3.2% | decrease ✓ |
| carB | raise_roof | −0.0379 | 4.1 | +0.7% | −4.3% | decrease ✓ |

**3 of 6 pairs are usable, and only 1 is drag-increasing** — the one the campaign
existed to produce. Frozen as `benchmarks/les_drag_pairs_v1.json`.

**The scoring method had to be fixed first, and it decided the verdict.** The result
file's `cd_std` is time-series scatter, not the error on the mean, and a 4-level grid
legitimately *raises* it by resolving more turbulence. The runner logs a cumulative
mean every 900 samples, so block means can be recovered by differencing and their
scatter gives a real standard error (`estimate_les_mean_error.py`). For carA the two
differ threefold — 0.0070 instantaneous vs **±0.0022** on the mean. Scored the naive
way, the campaign's only success would have been discarded as noise.

**Why the other two carA variants failed is now understood, and it is the useful
output.** The solver recomputes `area_ref = bbox width × height` per geometry
(`stl_loader.py`), so widening the car grows the denominator. carA/wide_rear raised
the drag **force** by 1.9% while the reference area grew 4.5% — Cd fell. The physics
worked; the normalisation swallowed it.

**So the design rule for raising Cd is: add drag force without changing the frontal
box — extend along x only.** carA/blunt_tail is exactly that (frontal area +0.0%,
force +3.3%, all of it landing in Cd). This also explains why the 400-pair G2 pool
holds only 3 drag-increasing pairs: most shape changes that add drag also add frontal
area, and the two cancel.

**But the recipe is not universal.** The same blunt_tail gives +3.3% force on carA and
−0.5% on carB. carB is a much lower body (1.05 m vs 1.32 m) where squaring off the
tail appears to tidy the wake rather than enlarge it. Good for a benchmark — a model
must actually read the shape — but it rules out "this deformation always increases
drag" as a generation strategy.

**Honest verdict: the bottleneck is not cleared.** 32 GPU-hours bought one
drag-increasing pair on unseen geometry, taking the total from 3 to 4. At this hit
rate (1 in 6), the ~10 pairs needed for statistical power would cost roughly 240
GPU-hours. A second campaign should first fix two things that cost this one dearly:
target x-extension only, and avoid the precision loss carB suffered — its SEM was 3×
carA's because the inflow had to drop to u_inf 0.08 for stability, which is better
addressed by rescaling the geometry than by slowing the flow.

---

### 2026-09-01 (2nd) — Retraction: there is no reference-area bookkeeping bug

Section 2026-08-27(3) reported, as a side finding, that "the reference area is wrong
by 3× on 3 gate labels" and concluded "this is a bookkeeping problem, not a model
problem." **That conclusion was wrong and is retracted.**

Sweeping all ~660 cases (`sweep_reference_area.py`) found 106 (16%) where the
declared `ref_area` disagrees with the bounding-box frontal area by more than 15%,
including 30 cases off by 2.5-9×. That looked like confirmation of a systematic bug.

It is not. `which_area_matches_label.py` settles it by integrating the stored surface
field and asking which divisor reproduces the recorded `su2_cd`:

| run | label Cd | declared A | measured A | ÷ declared | ÷ measured |
|---|---|---|---|---|---|
| 58 | 0.2147 | 5.897 | 1.863 | **0.2146 (0%)** | 0.6792 (216%) |
| 52 | 0.1807 | 4.470 | 1.488 | **0.1805 (0%)** | 0.5422 (200%) |
| 79 | 0.1956 | 4.470 | 1.437 | **0.1953 (0%)** | 0.6075 (211%) |
| 139 | 0.1440 | 5.412 | 1.047 | **0.1435 (0%)** | 0.7420 (415%) |
| 30 | 0.3441 | 0.214 | 0.068 | **0.3437 (0%)** | 1.0854 (215%) |

**The declared value reproduces the label to 0-2% on all 15 cases with a finite
label, including every one of the "3× wrong" ones.** Label and evaluator use the same
convention consistently. The declared reference is simply not the bounding-box
frontal area for these shapes — a legitimate choice for slender bodies.

**What the earlier analysis actually found, correctly stated**: on run_52/58/79 the
model under-predicts the drag **force** by 3-4×. These are slender geometries
(run_58 is 1.578 × 1.181 m in cross-section) far from the training distribution.
Multiplying the prediction by declared/measured happened to improve the gate score
because that ratio correlates with slenderness, which correlates with the error —
not because it corrected a label. **It is a model problem after all, and the
"±20% within 6/15 → 9/15" improvement reported at the time was an artefact of
applying a fudge factor.**

Consequences:
- Priority 2 from that section ("sweep the whole training pool for the 3× error") is
  **closed as unnecessary**. There is nothing to fix in the labels.
- The gate's worst cases are worst because the geometry is unusual, which is the same
  story as the shape-class split: the model does well on cars and poorly off them.

---

### 2026-09-02 — Sealing cascade lifted out as a standalone service

The cascade that turns a dirty STL into a watertight one lived inside the LES
runner's `stl_loader.py`, which imports jax at module scope — using it meant
pulling in the whole solver stack. It is now `aox_g3/seal.py` plus a CLI
(`scripts/seal_geometry.py`), depending on nothing beyond numpy and trimesh.
No solver wiring, by request.

**Three tiers, in order of how little they disturb the shape** — the
"reconstruct, don't repair" line of Portaneri et al. (ACM TOG 2022):

1. **OpenVDB level set** (`vdb_tool`), offset bisected per shape. Preferred because
   it closes gaps at a shorter distance than Alpha Wrap (24 mm vs 29 mm on the
   GT-R), keeping narrow features like a wing support alive.
2. **CGAL `alpha_wrap_3`** — takes a triangle soup, so it handles non-manifold input
   the other tiers cannot load at all.
3. **Voxel flood-fill seal** (`fix_shell`, Warp GPU distance field) — last, because
   its closing distance is coarse enough to bridge a wing to the body.

All measured constants were carried over with their provenance in comments
(alpha = diag/180, offset = alpha/30, volume-ratio floor 0.5, openness threshold 5.0).

**The design change that matters: a tier can no longer fail silently.**
`SealReport` always states which tools were available, which tiers were attempted,
and every warning. The v8 label campaign lost seven 4-hour LES runs because
`vdb_tool` was missing on the host, tier 1 fell through unnoticed, and the
open-shell result looked plausible until the run-to-run scatter came back 4x too
large. Running `--tools` on G3_TEST already shows the same class of gap today:
`vdb_tool` 있음, `fix_shell` 있음, **`cgal_alpha_wrap` 없음** — tier 2 would be
skipped there and nobody had noticed.

**Verified against a known answer.** `make_dirty_mesh.py` punches holes in a
watertight mesh so the seal can be checked rather than merely observed to produce
something closed. On carA_base with 140 holes (openness 0 → 7.76), tier 1 sealed it:

| | volume | area | watertight |
|---|---|---|---|
| original (truth) | 7.8547 | — | True |
| holed | 7.6533 | — | False |
| **sealed** | **7.8426 (−0.15%)** | **+0.67%** | **True** |

Extents came back within 5 mm on a 5 m car.

**A real bug found in the port**: the LES copy hard-codes "mm" in its offset
message, so a metre-scale mesh reports a 0.054 m bound as "0.1mm" and the operator
reads it as a broken parameter. The cascade itself is scale-invariant (everything
is a fraction of the diagonal); only the label was wrong. Fixed here.

**A usability trap fixed**: when openness is below the threshold the right answer is
to pass the mesh through untouched (a winding number handles it, and sealing would
inflate volume and fill narrow gaps for nothing). But the CLI was reporting that as
"OK" while writing a non-watertight file. It now distinguishes 수밀 from
통과(비수밀), and `--require-watertight` makes the pass-through an error for callers
that genuinely need a closed surface (exit code 1, verified).

---

### 2026-09-02 — STEP path proven on real CATIA output; the remaining gap is hidden-geometry removal

Judged the STEP route viable and built it, then ran a genuine BAIC-class file
through end to end: `CAS-A.stp`, 36 MB, header says **CATIA V5 STEP Exchange**.

**What works, measured** (`aox_g3/cad.py`, `aox_g3/symmetry.py`)

| Stage | Result |
|---|---|
| STEP read with names/colours (`cadquery-ocp`) | **7.3 s** |
| B-rep diagnosis | **1.7 s** — solids 0, shells 2,847, faces 2,847, **free edges 12,724 of 12,728**, invalid faces **0** |
| **B-rep sewing** (`BRepBuilderAPI_Sewing`, tol = diag×0.002) | shells **2,847 → 16**, free edges **12,724 → 1,203**, openness **168 → 17.7** |
| Symmetry detection + mirror | found **y=0 spanning 91% of the silhouette**, width **1,198 → 1,955 mm**, 4,743 seam vertices welded |
| Tessellation | 0.9 s, 68,543 triangles |

**Diagnosing before tessellating paid off immediately.** "Invalid faces 0 but 99.97%
of edges free" says the surfaces are fine and only the topology is missing, which
is what makes sewing the right move and rules out face repair. Sewing then removed
91% of the free edges — those neighbours were unstitched, not absent.

`cadquery-ocp` also settles a practical question: it installs from pip, unlike
`pythonocc-core` which is conda-only, so an OpenCascade pipeline can be deployed
the same way as everything else here.

**Where it stops: alpha wrap returns a shell, not a body.**

| alpha | watertight | volume | vs bounding box |
|---|---|---|---|
| diag/180 = 29.1 | True | 0.118 m³ | 0.8% |
| diag/110 = 47.7 | True | 0.361 m³ | 2.4% |
| diag/90 = 58.3 | True | 0.454 m³ | **3.0%** |

A car should fill 30-45%. Four hypotheses were tested and three eliminated:

1. **Gaps wider than alpha?** Measured: median 5.7 mm, 95th percentile 37.4 mm,
   max 474 mm (`measure_gaps.py`). Plausible at first — but sewing cut the free
   edges by 91% and the wrap's volume changed by less than a thousandth
   (0.119 → 0.118 m³). Not the cause.
2. **Half model?** Confirmed and fixed — but mirroring changed nothing either.
3. **Surface does not enclose a volume?** Ray-cast from 400 interior points:
   **92% odd crossings** (`check_enclosure.py`). It encloses a volume.
4. **The wrap is wrapping interior geometry.** Wrap area / input area = **1.60**
   (1.0 would be a solid, 2.0 a sheet wrapped both sides), and the input's
   30.4 m² is well above the 12-20 m² of a car's exterior skin. The model carries
   inner panels — wheel-arch liners, cabin, engine bay — and the wrap faithfully
   reaches them through the openings.

**So the missing step is hidden-geometry removal, exactly as specified in the
research note**: flood-fill from an outside seed, keep the reachable surface,
discard the rest. That is what Flexcompute's GeometryAI calls "Remove hidden
geometry" with a "Min passage size", and the machinery already exists here —
`fix_shell` does a voxel flood fill, it is just wired for sealing rather than for
classifying reachability.

**Corrected pipeline order**:

```
STEP ─▶ B-rep diagnose ─▶ sew ─▶ mirror ─▶ tessellate
                                              │
                                              ▼
                              remove hidden geometry  ← missing
                                              │
                                              ▼
                                   seal cascade ─▶ watertight solid
```

**One caveat on the volume test used throughout the seal cascade**: `_volume_ok`
compares the sealed result against the *input's* volume, and for a triangle soup
that number is a divergence-theorem artefact. It happened to be plausible here
(4.33 m³, 47% of the box) so the verdicts stand, but for a genuinely open input
the comparison should be against the bounding box instead.

**Hidden-geometry removal added, and the wrap still returns a shell**

`aox_g3/visibility.py` implements the missing step: voxelise, flood the empty space
from outside the box, keep the faces that region touches. Its one parameter is the
voxel pitch, which is Flexcompute's "Min passage size" under another name — a gap
narrower than the pitch reads as closed and whatever hides behind it is classified
interior. On CAS-A it runs in **0.6 s** and removes **80,731 of 137,086 faces**,
taking the surface from 30.4 m² to **10.5 m²** — the right order for a car's skin.

It did not fix the wrap. Wrapping the visible-only surface gives 0.3-1.7% of the
bounding box, slightly worse than before, because removing faces adds holes.

**The measurement everything rested on turned out to be unreliable, and checking it
changed the diagnosis.** Volumes had been read from `trimesh.volume`, which
integrates over faces and is meaningless when a surface leaks. Measuring instead by
voxel occupancy — flood from outside, count what is neither wall nor outside
(`scripts/voxel_volume.py`):

| mesh | divergence | **voxel** |
|---|---|---|
| carA_base (a solved G2 surface) | 60.6% | **55.8%** — the two agree, it is genuinely closed |
| cas_final (CAS-A, sewn + mirrored) | 59.1% | **0.0%** |
| alpha wrap of it | 3.0% | 0.5% |
| fix_shell of it | 5.5% | 1.3% |

**CAS-A's plausible-looking 59.1% is false.** A flood fill from outside reaches
every interior cell at a 21 mm pitch, so after sewing and mirroring the surface
still has holes wider than that. The wrap is a shell because its input leaks, and
both wrappers were right to be rejected.

So `_volume_ok` was fixed rather than the wrappers: for a **watertight** input it
still compares against the input's volume, but for an **open** one it now judges the
result on how much of its own bounding box it fills (floor 0.15, against the 55.8%
a real car measures). Verdicts are unchanged and now rest on a number that means
something — the known-good 140-hole test reports **0.61 fill** and passes, CAS-A
reports 0.01-0.09 and fails.

**Where this leaves the STEP path**: everything up to the wrap is verified on real
CATIA output, and the remaining blocker is specific and measurable — holes wider
than 21 mm that survive sewing. Finding and capping those is the next step, and the
tooling to locate them (`scripts/voxel_volume.py`, the openness metric) is in place.

## Returning STEP, not a mesh

The requirement to hand the CAD engineer back a STEP file rules out the mesh route
rather than adding a step to it. Its output is a triangle soup, and turning that
back into CAD means one planar face per triangle - a 137,000-face STEP no CAD
system opens - or refitting NURBS, which replaces the supplier's surfaces with an
approximation. So the repair now stays in B-rep from end to end (`aox_g3/brep.py`,
`scripts/heal_step.py`), and the mesh cascade in `aox_g3/seal.py` keeps the job it
is right for: STL and OBJ input, where there is no topology to preserve.

**Staying in B-rep also settled the question the mesh route could not.** Finding
the leak had come down to flood-filling from outside, noticing it reached every
interior cell, and having nothing to say about where. `ShapeAnalysis_FreeBounds`
answers directly, in **0.0 s**: 109 closed free boundaries, the largest **4.85 m**
across with a 16 m perimeter, against a 5.25 m car. No sealing tolerance measured
in millimetres was ever going to close that; it was never a tuning problem.

Sorted by size the boundaries name themselves - underbody, cabin opening, door
glass, wheel arches. **CAS-A is an exterior surface model, and what is missing is
missing by design, not damage.**

**The sewing tolerance was wrong, and the note justifying it was wrong.** `cad.py`
recorded a measured median gap of 5.7 mm and picked diagonal x 0.002 - 10.5 mm - to
clear it. Sweeping the tolerance shows the gaps are overwhelmingly sub-millimetre:
at 0.1 mm the shells already collapse 2,847 to 36 and the free edges 12,724 to
1,874. The 5.7 mm median had the genuine openings mixed in with the seams.

Coarsening costs real damage once the result has to leave as CAD, because OCC
stores its tolerance on every edge it merges:

| tolerance | shells | free edges | invalid faces | holes |
|---|---|---|---|---|
| 0.10 | 36 | 1,874 | 61 | 165 |
| 1.00 | 26 | 1,672 | **46** | 146 |
| 5.00 | 22 | 1,511 | 83 | 169 |
| 10.51 | 16 | 1,203 | 96 | 109 |

Surface area moves under 0.1% across the whole range, so nothing is reshaped
either way - the only question is how much topological slop lands in the file.

**Sewing in stages beats sewing once, on every axis at the same time.** Edges
joined at 1 mm are no longer free, so a later coarser pass can only reach what is
still open, and 10 mm of slop lands on the few hundred edges that need it instead
of all 12,724:

| stages | free edges | invalid faces | holes | largest perimeter | time |
|---|---|---|---|---|---|
| 1.05 | 1,591 | 33 | 140 | 35.2 m | 7.3 s |
| 10.51 | 1,203 | 96 | 109 | 16.0 m | 19.6 s |
| **1.05 → 5 → 10.51** | **765** | **46** | **56** | 15.7 m | **8.9 s** |

A third fewer free edges, half the invalid faces, half the holes, and less than
half the time. `cad.sew_progressive` is now the default ladder.

**The STEP round trip had to be verified, and it caught two real losses.** Writing
with pcurves on - OCC's default - produced a 106 MB file that read back **645 faces
short and 9.6% smaller in area**. Pcurves are recomputed on import by the receiving
system anyway, so they are off. With that and the staged sewing:

| | before | after |
|---|---|---|
| file | 106 MB | **35.2 MB** (input is 36.6) |
| faces written → read | 2,887 → 2,242 | **2,850 → 2,850** |
| area | 15.02 → 13.58 m² | **15.02 → 15.00 m²** |
| invalid faces | 130 | 33 |

**And the 15 colours are a mirage.** They exist in the palette and are assigned to
nothing - no face, no label - and the file has one part name over 2,847 faces. The
report was counting the palette. Since colours are how a CFD engineer picks
boundary patches, this is worth knowing: **on this file the patches have to come
from geometry, because the CAD does not carry them.**

**The surface filler cannot be trusted on its own word.** `BRepOffsetAPI_MakeFilling`
returns `IsDone` for boundaries it has not solved. Filling 42 holes took the model
from 15.02 m² to **191.44 m²** with nothing in the API objecting. Measuring each
patch against the hole it closes separates the two populations cleanly - a hole
spanning d closes with something of order d²:

- good: **0.08 – 1.15** (every planar patch at exactly 0.39)
- runaway: **1.90 – 204**, plus one with **negative area** (an inverted face)

The worst single patch was **155 m² for an 871 mm hole** - ten cars' worth of
surface. `MAX_PATCH_RATIO = 1.5` sits in the empty gap between the populations, and
a second check catches many small patches summing to a body that is no longer the
body that came in. With it: 32 of 46 holes filled, **15.02 → 17.43 m²**, and the 14
rejects reported with the reason.

**Where the pipeline stands.** STEP in, STEP out, verified round trip, holes
located and classified, with the ones that need a human decision listed by size and
position. What remains open on CAS-A is the underbody and the cabin - closing those
is a modelling decision about what the model should be, not a repair.
