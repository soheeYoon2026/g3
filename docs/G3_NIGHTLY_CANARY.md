# G3 nightly training and shadow canary

## Live topology

The Seoul GPU EC2 currently hosts both inference processes:

- `g3-inference.service`, port 8003: production response path
- `g3-challenger.service`, port 8004: localhost-only shadow candidate
- `g3-tunnel.service`: public tunnel to production port 8003
- `g3-nightly.timer`: starts the lifecycle at 00:00 Asia/Seoul, with up to
  five minutes of randomized delay

The EC2 must stay running while it owns the G3 inference endpoint.

## Data and model flow

```text
successful G1/G2/G4 job
  -> idempotent private S3 training-event JSON
  -> nightly event sync and quality gate
  -> wait until at least 10 new cases
  -> rebuild G2 fields and G1/G4 coefficient datasets
  -> exclude frozen validation cases
  -> train versioned challenger
  -> fixed offline G1/G2/G4 Cd/Cl gates
  -> stage models/registry/challenger.pt
  -> shadow 10% of production requests
  -> require 20 successful runtime comparisons
  -> promote models/registry/production.pt atomically
  -> retain models/registry/previous.pt for rollback
```

Failed solver runs never become regression targets. The backend event contains
S3 artifact references, numeric flow conditions, and solver summaries; it does
not copy arbitrary user configuration into the training inventory.

## Gates

- Start training only after 10 new successful job events.
- Candidate MAE may regress by at most 3% on every configured fixed holdout:
  G2 normal Cd/Cl, G2 high-drag Cd/Cl, G1 Cd/Cl, and G4 Cd.
- Shadow promotion requires at least 20 observations, error rate at most 1%,
  p95 absolute Cd and Cl drift at most 0.05, and p95 latency ratio at most 1.5.
- A candidate with enough shadow samples that fails a runtime gate is rejected.
- A pending candidate is not overwritten by another nightly training run.

## Operations

Status:

```bash
systemctl status g3-inference g3-challenger g3-tunnel g3-nightly.timer
systemctl list-timers g3-nightly.timer
python scripts/model_registry.py status
tail -f var/canary/shadow.jsonl
```

Manual rollback:

```bash
python scripts/model_registry.py rollback --reason "operator rollback"
systemctl restart g3-inference
```

The stage backend must be deployed with
`G3_TRAINING_CAPTURE_ENABLED=true`. Stage settings default to enabled;
production settings remain opt-in.

