"""VTK output and streamline helpers for field-surrogate inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _vtk():
    try:
        import vtk
        from vtk.util import numpy_support as nps
    except ImportError as exc:  # pragma: no cover
        raise ImportError("VTK output requires: pip install vtk") from exc
    return vtk, nps


def write_volume_vti(
    path: str | Path,
    dimensions: tuple[int, int, int],
    origin: np.ndarray,
    spacing: np.ndarray,
    cp: np.ndarray,
    velocity: np.ndarray,
    pressure: np.ndarray,
):
    vtk, nps = _vtk()
    image = vtk.vtkImageData()
    image.SetDimensions(*dimensions)
    image.SetOrigin(*map(float, origin))
    image.SetSpacing(*map(float, spacing))
    for name, values in (
        ("Pressure_Coefficient", cp),
        ("Pressure", pressure),
        ("Velocity", velocity),
        ("Speed", np.linalg.norm(velocity, axis=1)),
    ):
        array = nps.numpy_to_vtk(np.ascontiguousarray(values, dtype=np.float32), deep=True)
        array.SetName(name)
        image.GetPointData().AddArray(array)
    image.GetPointData().SetActiveVectors("Velocity")
    # Some VTK readers only expose the active scalar array on vtkImageData.
    # Keep Speed both named and active so downstream viewers see it reliably.
    image.GetPointData().SetActiveScalars("Speed")
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(image)
    writer.SetDataModeToBinary()
    writer.Write()
    return image


def write_surface_vtp(
    path: str | Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    cp: np.ndarray,
    pressure: np.ndarray,
    velocity: np.ndarray | None = None,
):
    vtk, nps = _vtk()
    poly = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    points.SetData(nps.numpy_to_vtk(np.ascontiguousarray(vertices, dtype=np.float32), deep=True))
    poly.SetPoints(points)
    cells = vtk.vtkCellArray()
    packed = np.column_stack([np.full(len(faces), 3, dtype=np.int64), faces]).ravel()
    cells.SetCells(len(faces), nps.numpy_to_vtkIdTypeArray(packed, deep=True))
    poly.SetPolys(cells)
    arrays = [
        ("Pressure_Coefficient", cp),
        ("Pressure", pressure),
        ("pMean", pressure),
        ("normalDisplacementFace", np.zeros_like(cp)),
    ]
    if velocity is not None:
        arrays.extend(
            (
                ("Velocity", velocity),
                ("UMean", velocity),
                ("Speed", np.linalg.norm(velocity, axis=1)),
            )
        )
    for name, values in arrays:
        array = nps.numpy_to_vtk(np.ascontiguousarray(values, dtype=np.float32), deep=True)
        array.SetName(name)
        poly.GetPointData().AddArray(array)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(poly)
    writer.SetDataModeToBinary()
    writer.Write()
    return poly


def write_streamlines(
    volume,
    path: str | Path,
    body_bounds: tuple[np.ndarray, np.ndarray],
    flow_direction: np.ndarray,
    resolution: int = 48,
):
    vtk, _ = _vtk()
    lo, hi = (np.asarray(body_bounds[0]), np.asarray(body_bounds[1]))
    span = hi - lo
    flow_axis = int(np.argmax(np.abs(flow_direction)))
    sign = 1.0 if flow_direction[flow_axis] >= 0 else -1.0
    cross = [axis for axis in range(3) if axis != flow_axis]
    rake_axis = 2 if 2 in cross else cross[0]
    other_axis = next(axis for axis in cross if axis != rake_axis)
    p1 = (lo + hi) / 2.0
    p2 = p1.copy()
    offset = 0.35 * max(span)
    p1[flow_axis] = lo[flow_axis] - offset if sign > 0 else hi[flow_axis] + offset
    p2[flow_axis] = p1[flow_axis]
    p1[other_axis] = p2[other_axis] = (lo[other_axis] + hi[other_axis]) / 2.0
    p1[rake_axis] = lo[rake_axis] + 0.05 * span[rake_axis]
    p2[rake_axis] = hi[rake_axis] + 0.25 * span[rake_axis]

    line = vtk.vtkLineSource()
    line.SetPoint1(*map(float, p1))
    line.SetPoint2(*map(float, p2))
    line.SetResolution(resolution)
    tracer = vtk.vtkStreamTracer()
    tracer.SetInputData(volume)
    tracer.SetSourceConnection(line.GetOutputPort())
    tracer.SetIntegratorTypeToRungeKutta45()
    tracer.SetIntegrationDirectionToForward()
    tracer.SetMaximumPropagation(8.0 * max(span))
    tracer.SetInitialIntegrationStep(0.02 * max(span))
    tracer.SetComputeVorticity(False)
    tracer.Update()
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(tracer.GetOutput())
    writer.SetDataModeToBinary()
    writer.Write()
    return tracer.GetOutput()


def render_png(surface_path: str | Path, streamlines_path: str | Path, output: str | Path):
    try:
        import pyvista as pv
    except ImportError as exc:  # pragma: no cover
        raise ImportError("PNG rendering requires: pip install pyvista") from exc
    plotter = pv.Plotter(off_screen=True, window_size=(1600, 900))
    surface = pv.read(surface_path)
    lines = pv.read(streamlines_path)
    plotter.add_mesh(surface, scalars="Pressure_Coefficient", cmap="coolwarm", smooth_shading=True)
    if lines.n_points:
        tube = lines.tube(radius=max(surface.length * 0.0015, 1e-6))
        # vtkTubeFilter may drop point-data arrays on sparse/short streamlines.
        # Check the transformed mesh rather than the source before coloring it.
        if tube.n_points:
            plotter.add_mesh(
                tube,
                scalars="Speed" if "Speed" in tube.point_data else None,
                cmap="viridis",
            )
    plotter.add_axes()
    plotter.view_isometric()
    bounds = np.asarray(surface.bounds, dtype=float).reshape(3, 2)
    span = float(np.max(bounds[:, 1] - bounds[:, 0]))
    focus_bounds = tuple(np.column_stack([
        bounds[:, 0] - 0.45 * span,
        bounds[:, 1] + 0.45 * span,
    ]).ravel())
    plotter.reset_camera(bounds=focus_bounds)
    plotter.show(screenshot=str(output), auto_close=True)
