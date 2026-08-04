"""Surrogate models: a torch-free baseline and the PointNet regressor."""

from .baseline_sklearn import PooledMLPRegressor

__all__ = ["PooledMLPRegressor"]

# PointNet is imported lazily (see get_pointnet) because torch is optional in
# the Phase 0 environment. The sklearn baseline must import with numpy only.


def get_pointnet(*args, **kwargs):
    """Import and construct the torch PointNet regressor on demand.

    Kept out of module import so ``import aox_g3.models`` works without torch.
    """
    from .pointnet import PointNetRegressor

    return PointNetRegressor(*args, **kwargs)
