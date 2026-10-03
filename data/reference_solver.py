"""Independent reference solvers for the lined-pipe beam on an elastic foundation.

The module contains two routes:
1. analytical Green-kernel superposition for a constant intact foundation;
2. a finite-difference solution of EI*w'''' + k(x)*w = q(x).

Sign convention:
    x rightward positive
    q and w downward positive
    theta = w'
    M = -EI*w''
    Q = M'
    Q' = k*w - q
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


FIELD_NAMES = ("w", "theta", "M", "Q")


@dataclass(frozen=True)
class BeamParameters:
    ei: float
    k0: float
    host_outer_diameter: float

    def __post_init__(self) -> None:
        if self.ei <= 0.0:
            raise ValueError("ei must be positive")
        if self.k0 <= 0.0:
            raise ValueError("k0 must be positive")
        if self.host_outer_diameter <= 0.0:
            raise ValueError("host_outer_diameter must be positive")

    @property
    def beta(self) -> float:
        return float((self.k0 / (4.0 * self.ei)) ** 0.25)

    @property
    def characteristic_length(self) -> float:
        return 1.0 / self.beta

    @property
    def outer_fiber_radius(self) -> float:
        return 0.5 * self.host_outer_diameter


def annulus_second_moment(d_outer: float, d_inner: float) -> float:
    if not d_outer > d_inner > 0.0:
        raise ValueError("diameters must satisfy d_outer > d_inner > 0")
    return float(np.pi * (d_outer**4 - d_inner**4) / 64.0)


def composite_ei(
    host_outer_diameter: float,
    host_thickness: float,
    host_modulus: float,
    liner_thickness: float,
    liner_modulus: float,
) -> float:
    d_host_inner = host_outer_diameter - 2.0 * host_thickness
    d_liner_outer = d_host_inner
    d_liner_inner = d_liner_outer - 2.0 * liner_thickness
    if min(host_thickness, liner_thickness, host_modulus, liner_modulus) <= 0.0:
        raise ValueError("thicknesses and moduli must be positive")
    i_host = annulus_second_moment(host_outer_diameter, d_host_inner)
    i_liner = annulus_second_moment(d_liner_outer, d_liner_inner)
    return float(host_modulus * i_host + liner_modulus * i_liner)


def uniform_grid(x_left: float, x_right: float, dx: float) -> np.ndarray:
    if not x_right > x_left:
        raise ValueError("x_right must exceed x_left")
    if dx <= 0.0:
        raise ValueError("dx must be positive")
    count = int(round((x_right - x_left) / dx))
    if count < 8:
        raise ValueError("grid needs at least nine nodes")
    return np.linspace(x_left, x_right, count + 1)


def midpoint_cells(
    x_left: float,
    x_right: float,
    requested_dx: float,
) -> tuple[np.ndarray, float]:
    if not x_right > x_left:
        raise ValueError("cell interval must have positive length")
    if requested_dx <= 0.0:
        raise ValueError("requested_dx must be positive")
    count = max(1, int(np.ceil((x_right - x_left) / requested_dx)))
    actual_dx = (x_right - x_left) / count
    centers = x_left + (np.arange(count, dtype=float) + 0.5) * actual_dx
    return centers, actual_dx


def top_hat_nodal_load(
    x: np.ndarray,
    *,
    center: float,
    width: float,
    total_force: float,
) -> np.ndarray:
    """Return a nodal top-hat load rescaled to exact trapezoidal total force."""

    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("x must be a one-dimensional grid")
    if width <= 0.0 or total_force <= 0.0:
        raise ValueError("width and total_force must be positive")
    mask = np.abs(x - center) <= 0.5 * width + 8.0 * np.finfo(float).eps
    q = np.zeros_like(x)
    q[mask] = total_force / width
    represented = float(np.trapezoid(q, x))
    if represented <= 0.0:
        raise ValueError("grid does not resolve the load footprint")
    q *= total_force / represented
    return q


def support_profile(
    x: np.ndarray,
    *,
    k0: float,
    retention: np.ndarray | float,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    retention_array = np.broadcast_to(np.asarray(retention, dtype=float), x.shape)
    if np.any((retention_array < 0.0) | (retention_array > 1.0)):
        raise ValueError("retention must lie in [0, 1]")
    if k0 <= 0.0:
        raise ValueError("k0 must be positive")
    return k0 * retention_array


def green_point_kernels(
    separation: np.ndarray,
    *,
    beta: float,
    k0: float,
) -> dict[str, np.ndarray]:
    if beta <= 0.0 or k0 <= 0.0:
        raise ValueError("beta and k0 must be positive")
    separation = np.asarray(separation, dtype=float)
    radius = np.abs(separation)
    z = beta * radius
    decay = np.exp(-z)
    cosine = np.cos(z)
    sine = np.sin(z)
    coordinate_scale = max(float(np.max(radius, initial=0.0)), 1.0)
    tolerance = 16.0 * np.finfo(float).eps * coordinate_scale
    sign = np.where(radius <= tolerance, 0.0, np.sign(separation))
    return {
        "w": beta / (2.0 * k0) * decay * (cosine + sine),
        "theta": -sign * beta**2 / k0 * decay * sine,
        "M": 1.0 / (4.0 * beta) * decay * (cosine - sine),
        "Q": -0.5 * sign * decay * cosine,
    }


def green_uniform_load(
    x: np.ndarray,
    *,
    parameters: BeamParameters,
    center: float,
    width: float,
    total_force: float,
    source_dx: float,
) -> dict[str, np.ndarray | float]:
    x = np.asarray(x, dtype=float)
    sources, actual_dx = midpoint_cells(
        center - 0.5 * width,
        center + 0.5 * width,
        source_dx,
    )
    line_intensity = total_force / width
    cell_force = line_intensity * actual_dx
    fields = {name: np.zeros_like(x) for name in FIELD_NAMES}
    for start in range(0, sources.size, 128):
        block = sources[start : start + 128]
        kernels = green_point_kernels(
            x[:, None] - block[None, :],
            beta=parameters.beta,
            k0=parameters.k0,
        )
        for name in FIELD_NAMES:
            fields[name] += np.sum(kernels[name] * cell_force, axis=1)
    fields["strain_outer"] = outer_surface_strain(
        np.asarray(fields["M"]),
        ei=parameters.ei,
        signed_radius=parameters.outer_fiber_radius,
    )
    fields["foundation_reaction"] = parameters.k0 * np.asarray(fields["w"])
    fields["actual_source_dx"] = actual_dx
    fields["source_cells"] = float(sources.size)
    return fields


def _require_uniform_grid(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 9:
        raise ValueError("x must contain at least nine points")
    differences = np.diff(x)
    dx = float(differences[0])
    if dx <= 0.0 or not np.allclose(differences, dx, rtol=1e-11, atol=1e-13):
        raise ValueError("finite-difference grid must be uniformly increasing")
    return dx


def first_derivative(values: np.ndarray, dx: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    result = np.gradient(values, dx, edge_order=2)
    if values.size >= 7:
        result[2:-2] = (
            values[:-4]
            - 8.0 * values[1:-3]
            + 8.0 * values[3:-1]
            - values[4:]
        ) / (12.0 * dx)
    return result


def second_derivative(values: np.ndarray, dx: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    first = np.gradient(values, dx, edge_order=2)
    result = np.gradient(first, dx, edge_order=2)
    if values.size >= 7:
        result[2:-2] = (
            -values[:-4]
            + 16.0 * values[1:-3]
            - 30.0 * values[2:-2]
            + 16.0 * values[3:-1]
            - values[4:]
        ) / (12.0 * dx**2)
    return result


def finite_difference_response(
    x: np.ndarray,
    *,
    q: np.ndarray,
    ei: float,
    k: np.ndarray | float,
    outer_fiber_radius: float,
) -> dict[str, np.ndarray]:
    """Solve EI*w'''' + k(x)*w = q with far-field clamped-zero ends."""

    x = np.asarray(x, dtype=float)
    q = np.asarray(q, dtype=float)
    dx = _require_uniform_grid(x)
    if q.shape != x.shape:
        raise ValueError("q must have the same shape as x")
    if ei <= 0.0 or outer_fiber_radius <= 0.0:
        raise ValueError("ei and outer_fiber_radius must be positive")
    k_array = np.broadcast_to(np.asarray(k, dtype=float), x.shape)
    if np.any(k_array < 0.0):
        raise ValueError("foundation stiffness cannot be negative")

    n = x.size
    matrix = lil_matrix((n, n), dtype=float)
    rhs = np.zeros(n, dtype=float)

    matrix[0, 0] = 1.0
    matrix[1, 0:3] = (-3.0, 4.0, -1.0)
    for index in range(2, n - 2):
        matrix[index, index - 2 : index + 3] = (
            1.0,
            -4.0,
            6.0 + dx**4 * k_array[index] / ei,
            -4.0,
            1.0,
        )
        rhs[index] = dx**4 * q[index] / ei
    matrix[n - 2, n - 3 : n] = (1.0, -4.0, 3.0)
    matrix[n - 1, n - 1] = 1.0

    w = np.asarray(spsolve(matrix.tocsr(), rhs), dtype=float)
    theta = first_derivative(w, dx)
    moment = -ei * second_derivative(w, dx)
    shear = first_derivative(moment, dx)
    strain = outer_surface_strain(
        moment,
        ei=ei,
        signed_radius=outer_fiber_radius,
    )
    reaction = k_array * w
    return {
        "w": w,
        "theta": theta,
        "M": moment,
        "Q": shear,
        "strain_outer": strain,
        "foundation_reaction": reaction,
    }


def outer_surface_strain(
    moment: np.ndarray,
    *,
    ei: float,
    signed_radius: float,
) -> np.ndarray:
    if ei <= 0.0:
        raise ValueError("ei must be positive")
    return np.asarray(moment, dtype=float) * signed_radius / ei


def relative_l2(prediction: np.ndarray, reference: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=float)
    reference = np.asarray(reference, dtype=float)
    denominator = float(np.linalg.norm(reference))
    if denominator <= 0.0:
        raise ValueError("reference norm must be positive")
    return float(np.linalg.norm(prediction - reference) / denominator)


def normalized_parity_error(values: np.ndarray, parity: str) -> float:
    values = np.asarray(values, dtype=float)
    scale = max(float(np.max(np.abs(values))), np.finfo(float).tiny)
    if parity == "even":
        difference = values - values[::-1]
    elif parity == "odd":
        difference = values + values[::-1]
    else:
        raise ValueError("parity must be 'even' or 'odd'")
    return float(np.max(np.abs(difference)) / scale)


def equilibrium_error(
    x: np.ndarray,
    *,
    reaction: np.ndarray,
    q: np.ndarray,
) -> float:
    applied = float(np.trapezoid(q, x))
    resisted = float(np.trapezoid(reaction, x))
    if abs(applied) <= 0.0:
        raise ValueError("applied load must be nonzero")
    return abs(resisted - applied) / abs(applied)

