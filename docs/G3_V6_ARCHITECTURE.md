# G3 v6 architecture

```mermaid
flowchart LR
    subgraph Sources[Solver data sources]
        G1[G1 / OpenFOAM<br/>surface Cp, velocity, Cd]
        G2[G2 / SU2<br/>volume + surface fields, Cd, Cl]
        G4[G4 / LBM<br/>voxel fields, Cd]
        SMOKE[Smoke report<br/>8 test geometries / 44 run rows]
    end

    subgraph Ingestion[Ingestion and evidence]
        DEDUP[Deduplicate<br/>test + geometry + retry]
        EXTRACT[Extract labels from<br/>history.csv / results.json]
        QUALITY{Quality gate}
        FIELDONLY[Field-only sample<br/>coefficient label removed]
        FAIL[Failure / OOD evidence<br/>not a regression target]
    end

    subgraph Model[Shared G3 field model]
        STL[STL surface points<br/>XYZ + normals]
        COND[Flow conditions<br/>U, rho, mu, T, Lref, Aref]
        GE[Geometry encoder<br/>PointNet latent]
        CE[Condition encoder]
        FD[Implicit field decoder<br/>Cp + Ux/Uref + Uy/Uref + Uz/Uref]
    end

    subgraph Experts[Explicit coefficient domains]
        E2[G2 SU2 clean expert<br/>primary normal-range Cd + signed Cl]
        EH[G2 SU2 high-drag expert<br/>valid Cd > 0.5 / explicit selection]
        E1[G1 OpenFOAM expert<br/>separate calibration]
        E4[G4 LBM expert<br/>separate calibration]
        OOD[Per-expert latent centroid<br/>and q95 OOD radius]
    end

    subgraph Serving[Inference and verification]
        SELECT[Explicit expert selection<br/>default: g2_su2_clean]
        CHECK{OOD score <= 1?}
        RESULT[Return fields + Cd/Cl<br/>with expert and confidence]
        FALLBACK[Run G2 verification<br/>append verified case]
    end

    G1 --> DEDUP
    G2 --> DEDUP
    G4 --> DEDUP
    SMOKE --> DEDUP
    DEDUP --> EXTRACT --> QUALITY
    QUALITY -->|valid solver output<br/>surface points >= 5000| STL
    QUALITY -->|low mesh resolution| FIELDONLY --> FD
    QUALITY -->|failed run| FAIL
    STL --> GE
    COND --> CE
    GE --> FD
    CE --> FD
    GE --> E2
    CE --> E2
    GE --> EH
    CE --> EH
    GE --> E1
    CE --> E1
    GE --> E4
    CE --> E4
    GE --> OOD
    E2 --> SELECT
    EH --> SELECT
    E1 --> SELECT
    E4 --> SELECT
    OOD --> CHECK
    SELECT --> CHECK
    CHECK -->|yes| RESULT
    CHECK -->|no| FALLBACK --> G2
```

## Training policy

1. The Cp/velocity backbone may use every readable field sample.
2. Cd/Cl supervision is accepted only when the run succeeded and the surface
   resolution passes the quality gate. Low-resolution labels remain recorded
   as rejected evidence but do not contribute to coefficient loss.
3. G1, G2, and G4 coefficient values are never blended into one head. Their
   solver setup and reference normalization can produce different values for
   the same STL, so each label domain owns independent heads and statistics.
4. The production default is `g2_su2_clean`. G1/G4 experts must be selected
   explicitly when their calibration is desired.
5. Valid G2 cases above Cd 0.5 are kept out of the normal-range head and train
   `g2_su2_high_drag`. This routes by an explicit domain/OOD decision at
   inference, never by looking at an unknown true Cd.
6. Every expert stores its geometry-latent centroid and 95% training radius.
   A normalized OOD score above 1 triggers a G2 verification recommendation.
7. Smoke-test failures train or evaluate failure/OOD handling only; they are
   never treated as Cd/Cl regression labels.

## Current v6 data flow

```text
g2_fields_v5 (113 cases)
    + smoke successful G2 FINAL/RBF_DSN cases
    -> deduplicate by surface-flow artifact
    -> keep all valid fields
    -> coefficient gate: >= 5,000 surface points
    -> group split by geometry family/test case
    -> train g2_su2_clean field checkpoint
    -> attach separately calibrated G1/G4 coefficient experts
    -> evaluate by held-out group and smoke-test geometry
```
