"""AOX G3 — AI-model-based aerodynamic solver tier (Phase 0 spike).

G3 is the neural-surrogate engine for the AOX platform: it predicts drag/lift
coefficients (and, in the production model, surface pressure fields) directly
from an uploaded STL in seconds, instead of running an iterative PDE solve.

This package is the Phase 0 scaffold. It proves the end-to-end data path
(STL -> point cloud + SDF -> features -> Cd/Cl regression) on synthetic data
that runs today, and leaves clean seams for:
  * the production point-based operator (DoMINO / GINO) — see ``g3/domino``
  * label generation from the existing G1/G2/G4 solvers — see
    ``aox_g3.data.label_interface``

Nothing here is the production model. It is the smallest thing that lets us
smoke-test the pipeline and wire it into the FFD/morph optimisation loop.
"""

__version__ = "0.0.1"
