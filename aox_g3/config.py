"""Shared configuration and conventions for the G3 spike.

Keeping the geometry-processing knobs in one place matters: the STL sampling
used at *training* time and at *inference* time must be identical, or the model
sees a different input distribution than it was trained on. Import from here on
both paths rather than passing magic numbers around.
"""

from dataclasses import dataclass


# Objective columns the surrogate predicts, in fixed order. Cd first because it
# is the primary AOX objective; Cl second (downforce / lift). Extend with care —
# the model head width and every saved checkpoint depend on this ordering.
TARGETS = ("cd", "cl")


@dataclass(frozen=True)
class SampleConfig:
    """How an STL becomes a fixed-size point cloud for the surrogate."""

    n_surface_points: int = 4096      # points sampled on the surface (area-weighted)
    include_normals: bool = True      # feed per-point surface normals alongside xyz
    normalize: bool = True            # center + isotropically scale to a unit box
    seed: int = 0                     # sampling RNG seed (reproducibility)


# Padding of the FFD lattice beyond the geometry bounding box, as a fraction of
# the box size, so control points sit slightly outside the surface.
FFD_PAD = 0.05

DEFAULT_SAMPLE = SampleConfig()
