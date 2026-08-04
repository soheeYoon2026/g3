# G3 v6 quality-gated multi-expert field surrogate status

## Ready outputs

The inference pipeline writes these files for each STL:

1. `surface_pressure.vtp` — surface `Pressure` and `Pressure_Coefficient`
2. `volume_field.vti` — Cartesian volume `Velocity`, `Speed`, `Pressure`, and `Pressure_Coefficient`
3. `streamlines.vtp` — streamlines integrated from the predicted 3-D velocity field
4. `pressure_streamlines.png` — quick pressure/streamline preview
5. `prediction.json` — paths, device, pressure range, and speed range

## Current model

- Checkpoint: `models/g3_field_g2_v6_final.pt`
- Field cases: 149 G2 cases; 87 quality-gated coefficient cases
- Field backbone best epoch: 65 / 250
- Validation Cp MAE: 0.071897
- Validation velocity RMSE: 0.128220 Uref
- Default coefficient expert: `g2_su2_clean`
- Normal-range G2 validation Cd MAE / RMSE: 0.008844 / 0.012991
- Normal-range G2 validation Cl MAE / RMSE: 0.026485 / 0.027113
- Additional experts: `g2_su2_high_drag` (experimental), `g1_openfoam`,
  and `g4_lbm` (experimental)
- Every response carries the selected expert, OOD score, distribution flag,
  deployment status, and any solver-verification warning.

## Example result

The checked RS6 result is under `predictions/rs6_field_g1_v2/` and uses a
`96 x 64 x 48` volume grid at 30 m/s.

```bash
python -m aox_g3.infer_fields \
  --stl /path/to/body.stl \
  --model models/g3_field_g2_v6_final.pt \
  --out-dir predictions/my_case \
  --grid 96 64 48 \
  --u-x 30 \
  --density 1.225 \
  --viscosity 1.7894e-5 \
  --temperature 288.15 \
  --ref-length 5.0 \
  --ref-area 3.380196 \
  --png
```

Use `volume_field.vti`, `surface_pressure.vtp`, and `streamlines.vtp` directly
in ParaView. This model is suitable for rapid visualization and relative design
screening; it is not yet a replacement for quantitative CFD validation because
the volume-field supervision still contains only 29 G2 cases. G1 contributes
surface `pMean`, surface `UMean`, and Cd, but does not contain a volume field.

## Real-time service

The admin-only AOX preview calls the authenticated FastAPI service through the
Django BFF. Start the GPU service with:

```bash
G3_MODEL_PATH=models/g3_field_g2_v6_final.pt \
G3_COEFFICIENT_EXPERT=g2_su2_clean \
G3_API_TOKEN_FILE=/secure/path/g3-token \
uvicorn aox_g3.service:app --host 127.0.0.1 --port 8003
```

Django needs `G3_INFERENCE_URL` and `G3_INFERENCE_TOKEN`. The browser never
receives either value. Uploaded ASCII STL files are gzip-compressed by Django
before forwarding, and the service accepts up to 250 MiB after decompression.

On the current GPU instance, `g3-inference.service` keeps the API alive and
`g3-tunnel.service` publishes every newly assigned tunnel URL plus its token to
the private `_private/g3/inference-service.json` S3 object. Django reloads that
configuration every 60 seconds, so a tunnel restart needs no frontend change.
