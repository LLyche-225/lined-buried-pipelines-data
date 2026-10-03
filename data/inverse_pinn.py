"""Nondimensional mixed-state PINN components for distributed support inversion."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np
import torch
from torch import nn


STATE_NAMES = ("w", "theta", "M", "Q")
PHYSICS_NAMES = ("r_w", "r_theta", "r_M", "weak_Q")


@dataclass(frozen=True)
class PhysicalScales:
    length_m: float
    load_N_per_m: float
    deflection_m: float
    rotation: float
    moment_N_m: float
    shear_N: float
    strain: float
    domain_half_length_m: float

    @property
    def lambda_domain(self) -> float:
        return self.domain_half_length_m / self.length_m

    def to_dict(self) -> dict[str, float]:
        return asdict(self) | {"lambda_domain": self.lambda_domain}


def physical_scales(engineering: dict) -> PhysicalScales:
    ei = float(engineering["EI_N_m2"])
    k0 = float(engineering["k0_N_m2"])
    force = float(engineering["load_force_N"])
    width = float(engineering["load_width_m"])
    outer_radius = 0.5 * float(engineering["host_outer_diameter_m"])
    domain_half = float(engineering["domain_half_length_m"])
    if min(ei, k0, force, width, outer_radius, domain_half) <= 0.0:
        raise ValueError("engineering scales must be positive")
    length = (4.0 * ei / k0) ** 0.25
    load = force / width
    deflection = load / k0
    moment = ei * deflection / length**2
    return PhysicalScales(
        length_m=length,
        load_N_per_m=load,
        deflection_m=deflection,
        rotation=deflection / length,
        moment_N_m=moment,
        shear_N=ei * deflection / length**3,
        strain=moment * outer_radius / ei,
        domain_half_length_m=domain_half,
    )


class MixedStateNetwork(nn.Module):
    """Tanh state network with exact far-field displacement and rotation BCs."""

    def __init__(self, hidden_layers: int, width: int) -> None:
        super().__init__()
        if hidden_layers < 1 or width < 2:
            raise ValueError("invalid state-network size")
        layers: list[nn.Module] = []
        input_width = 1
        for _ in range(hidden_layers):
            linear = nn.Linear(input_width, width)
            nn.init.xavier_normal_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.extend((linear, nn.Tanh()))
            input_width = width
        output = nn.Linear(input_width, 4)
        nn.init.xavier_normal_(output.weight)
        nn.init.zeros_(output.bias)
        layers.append(output)
        self.network = nn.Sequential(*layers)

    def forward(self, zeta: torch.Tensor) -> torch.Tensor:
        raw = self.network(zeta)
        boundary_factor = 1.0 - zeta.square()
        displacement = boundary_factor.square() * raw[:, 0:1]
        rotation = boundary_factor * raw[:, 1:2]
        return torch.cat((displacement, rotation, raw[:, 2:3], raw[:, 3:4]), dim=1)


def open_uniform_knots(
    basis_count: int,
    degree: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if degree < 1 or basis_count <= degree:
        raise ValueError("basis_count must exceed the spline degree")
    interior_count = basis_count - degree - 1
    if interior_count:
        interior = torch.linspace(
            0.0,
            1.0,
            interior_count + 2,
            dtype=dtype,
            device=device,
        )[1:-1]
    else:
        interior = torch.empty(0, dtype=dtype, device=device)
    return torch.cat(
        (
            torch.zeros(degree + 1, dtype=dtype, device=device),
            interior,
            torch.ones(degree + 1, dtype=dtype, device=device),
        )
    )


def bspline_basis(
    coordinate_01: torch.Tensor,
    knots: torch.Tensor,
    degree: int,
) -> torch.Tensor:
    """Evaluate an open-uniform B-spline basis with Cox–de Boor recursion."""

    coordinate = coordinate_01.reshape(-1, 1)
    epsilon = 16.0 * torch.finfo(coordinate.dtype).eps
    coordinate = torch.clamp(coordinate, min=0.0, max=1.0 - epsilon)
    basis = (
        (coordinate >= knots[:-1].reshape(1, -1))
        & (coordinate < knots[1:].reshape(1, -1))
    ).to(coordinate.dtype)
    for order in range(1, degree + 1):
        new_count = basis.shape[1] - 1
        left_denominator = knots[order : order + new_count] - knots[:new_count]
        right_denominator = (
            knots[order + 1 : order + 1 + new_count]
            - knots[1 : 1 + new_count]
        )
        left_weight = torch.where(
            left_denominator > 0.0,
            (coordinate - knots[:new_count].reshape(1, -1))
            / torch.where(
                left_denominator > 0.0,
                left_denominator,
                torch.ones_like(left_denominator),
            ).reshape(1, -1),
            torch.zeros((coordinate.shape[0], new_count), dtype=coordinate.dtype, device=coordinate.device),
        )
        right_weight = torch.where(
            right_denominator > 0.0,
            (knots[order + 1 : order + 1 + new_count].reshape(1, -1) - coordinate)
            / torch.where(
                right_denominator > 0.0,
                right_denominator,
                torch.ones_like(right_denominator),
            ).reshape(1, -1),
            torch.zeros((coordinate.shape[0], new_count), dtype=coordinate.dtype, device=coordinate.device),
        )
        basis = left_weight * basis[:, :new_count] + right_weight * basis[:, 1 : new_count + 1]
    return basis


def smooth_support_envelope(
    x_m: torch.Tensor,
    *,
    core_half_width_m: float,
    outer_half_width_m: float,
) -> torch.Tensor:
    if not 0.0 < core_half_width_m < outer_half_width_m:
        raise ValueError("support envelope widths must satisfy 0 < core < outer")
    distance = torch.abs(x_m)
    transition = (outer_half_width_m - distance) / (
        outer_half_width_m - core_half_width_m
    )
    transition = torch.clamp(transition, 0.0, 1.0)
    smootherstep = transition**3 * (10.0 - 15.0 * transition + 6.0 * transition**2)
    return torch.where(distance <= core_half_width_m, torch.ones_like(x_m), smootherstep)


class BSplineSupportField(nn.Module):
    """Bounded support-loss spline with an exactly intact far field."""

    def __init__(
        self,
        *,
        basis_count: int,
        degree: int,
        core_half_width_m: float,
        outer_half_width_m: float,
        initial_loss_logit: float,
    ) -> None:
        super().__init__()
        self.basis_count = int(basis_count)
        self.degree = int(degree)
        self.core_half_width_m = float(core_half_width_m)
        self.outer_half_width_m = float(outer_half_width_m)
        self.coefficients = nn.Parameter(torch.zeros(self.basis_count))
        self.bias = nn.Parameter(torch.tensor(float(initial_loss_logit)))

    def forward(self, x_m: torch.Tensor) -> torch.Tensor:
        knots = open_uniform_knots(
            self.basis_count,
            self.degree,
            dtype=x_m.dtype,
            device=x_m.device,
        )
        coordinate = (x_m + self.outer_half_width_m) / (2.0 * self.outer_half_width_m)
        basis = bspline_basis(coordinate, knots, self.degree)
        logit = self.bias.to(x_m.dtype) + basis @ self.coefficients.to(x_m.dtype).reshape(-1, 1)
        envelope = smooth_support_envelope(
            x_m,
            core_half_width_m=self.core_half_width_m,
            outer_half_width_m=self.outer_half_width_m,
        )
        support_loss = envelope * torch.sigmoid(logit)
        return 1.0 - support_loss


class SingleRectangleSupportField(nn.Module):
    """Differentiable single-rectangle comparator with bounded parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.raw_center = nn.Parameter(torch.tensor(0.0))
        self.raw_half_width = nn.Parameter(torch.tensor(-1.715))
        self.raw_minimum_retention = nn.Parameter(torch.tensor(0.0))
        self.center_limit_m = 8.0
        self.minimum_half_width_m = 0.10
        self.maximum_half_width_m = 6.0
        self.edge_sharpness_per_m = 12.0

    def physical_parameters(self) -> dict[str, torch.Tensor]:
        center = self.center_limit_m * torch.tanh(self.raw_center)
        half_width = self.minimum_half_width_m + (
            self.maximum_half_width_m - self.minimum_half_width_m
        ) * torch.sigmoid(self.raw_half_width)
        minimum_retention = torch.sigmoid(self.raw_minimum_retention)
        return {
            "center_m": center,
            "half_width_m": half_width,
            "minimum_retention": minimum_retention,
        }

    def forward(self, x_m: torch.Tensor) -> torch.Tensor:
        values = self.physical_parameters()
        center = values["center_m"]
        half_width = values["half_width_m"]
        left = torch.sigmoid(
            self.edge_sharpness_per_m * (x_m - (center - half_width))
        )
        right = torch.sigmoid(
            self.edge_sharpness_per_m * (x_m - (center + half_width))
        )
        window = left - right
        return 1.0 - (1.0 - values["minimum_retention"]) * window


class InverseMixedPINN(nn.Module):
    """Load-specific mixed state networks coupled through one shared support field."""

    def __init__(self, load_state_count: int, model_config: dict) -> None:
        super().__init__()
        if load_state_count < 1:
            raise ValueError("at least one load state is required")
        self.state_networks = nn.ModuleList(
            MixedStateNetwork(
                hidden_layers=int(model_config["hidden_layers"]),
                width=int(model_config["width"]),
            )
            for _ in range(load_state_count)
        )
        representation = model_config["representation"]
        if representation == "bspline":
            self.support_field: nn.Module = BSplineSupportField(
                basis_count=int(model_config["spline_basis_count"]),
                degree=int(model_config["spline_degree"]),
                core_half_width_m=float(model_config["support_core_half_width_m"]),
                outer_half_width_m=float(model_config["support_outer_half_width_m"]),
                initial_loss_logit=float(model_config["initial_loss_logit"]),
            )
        elif representation == "single_rectangle":
            self.support_field = SingleRectangleSupportField()
        else:
            raise ValueError(f"unsupported representation: {representation}")
        self.representation = representation

    def state(self, state_index: int, zeta: torch.Tensor) -> torch.Tensor:
        return self.state_networks[state_index](zeta)

    def retention(self, zeta: torch.Tensor, domain_half_length_m: float) -> torch.Tensor:
        return self.support_field(float(domain_half_length_m) * zeta)


def derivative(values: torch.Tensor, coordinate: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        values,
        coordinate,
        grad_outputs=torch.ones_like(values),
        create_graph=True,
        retain_graph=True,
    )[0]


def normalized_top_hat_load(
    x_m: torch.Tensor,
    *,
    center_m: float,
    width_m: float,
) -> torch.Tensor:
    return (torch.abs(x_m - float(center_m)) <= 0.5 * float(width_m)).to(x_m.dtype)


def pointwise_residuals(
    model: InverseMixedPINN,
    state_index: int,
    zeta: torch.Tensor,
    *,
    load_center_m: float,
    engineering: dict,
    scales: PhysicalScales,
) -> dict[str, torch.Tensor]:
    zeta = zeta.requires_grad_(True)
    states = model.state(state_index, zeta)
    displacement, rotation, moment, shear = (
        states[:, index : index + 1] for index in range(4)
    )
    inverse_lambda = 1.0 / scales.lambda_domain
    x_m = float(engineering["domain_half_length_m"]) * zeta
    retention = model.retention(zeta, float(engineering["domain_half_length_m"]))
    normalized_load = normalized_top_hat_load(
        x_m,
        center_m=load_center_m,
        width_m=float(engineering["load_width_m"]),
    )
    return {
        "r_w": inverse_lambda * derivative(displacement, zeta) - rotation,
        "r_theta": inverse_lambda * derivative(rotation, zeta) + moment,
        "r_M": inverse_lambda * derivative(moment, zeta) - shear,
        "strong_Q": inverse_lambda * derivative(shear, zeta)
        - 4.0 * (retention * displacement - normalized_load),
    }


def _sample_x(
    count: int,
    *,
    load_centers_m: list[float],
    engineering: dict,
    sampling: dict,
    generator: torch.Generator,
    dtype: torch.dtype,
) -> torch.Tensor:
    if count < 8:
        raise ValueError("sample batch is too small")
    domain_half = float(engineering["domain_half_length_m"])
    global_count = int(round(count * float(sampling["global_fraction"])))
    local_count = count - global_count
    global_x = -domain_half + 2.0 * domain_half * torch.rand(
        (global_count, 1), generator=generator, dtype=dtype
    )
    if local_count:
        indices = torch.randint(
            0,
            len(load_centers_m),
            (local_count,),
            generator=generator,
        )
        centers = torch.as_tensor(load_centers_m, dtype=dtype)[indices].reshape(-1, 1)
        local_half = 0.5 * float(engineering["load_width_m"]) + float(
            sampling["load_local_margin_m"]
        )
        local_x = centers + local_half * (
            2.0 * torch.rand((local_count, 1), generator=generator, dtype=dtype) - 1.0
        )
        local_x = torch.clamp(local_x, -domain_half, domain_half)
        x_m = torch.cat((global_x, local_x), dim=0)
    else:
        x_m = global_x
    permutation = torch.randperm(x_m.shape[0], generator=generator)
    return x_m[permutation]


def sample_collocation(
    count: int,
    *,
    load_centers_m: list[float],
    engineering: dict,
    sampling: dict,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    x_m = _sample_x(
        count,
        load_centers_m=load_centers_m,
        engineering=engineering,
        sampling=sampling,
        generator=generator,
        dtype=dtype,
    )
    return (x_m / float(engineering["domain_half_length_m"])).to(device)


def sample_weak_intervals(
    count: int,
    *,
    load_centers_m: list[float],
    engineering: dict,
    sampling: dict,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    width = float(sampling["weak_interval_width_m"])
    domain_half = float(engineering["domain_half_length_m"])
    if not 0.0 < width < 2.0 * domain_half:
        raise ValueError("invalid weak interval width")
    centers = _sample_x(
        count,
        load_centers_m=load_centers_m,
        engineering=engineering,
        sampling=sampling,
        generator=generator,
        dtype=dtype,
    )
    centers = torch.clamp(
        centers,
        -domain_half + 0.5 * width,
        domain_half - 0.5 * width,
    )
    return (centers - 0.5 * width).to(device), (centers + 0.5 * width).to(device)


@lru_cache(maxsize=8)
def gauss_legendre(order: int) -> tuple[np.ndarray, np.ndarray]:
    if order < 2:
        raise ValueError("weak quadrature order must be at least two")
    nodes, weights = np.polynomial.legendre.leggauss(order)
    return nodes.astype(float), weights.astype(float)


def weak_equilibrium_residual(
    model: InverseMixedPINN,
    state_index: int,
    interval_left_m: torch.Tensor,
    interval_right_m: torch.Tensor,
    *,
    load_center_m: float,
    engineering: dict,
    scales: PhysicalScales,
    quadrature_order: int,
) -> torch.Tensor:
    if interval_left_m.shape != interval_right_m.shape:
        raise ValueError("weak interval endpoints must have matching shapes")
    if torch.any(interval_right_m <= interval_left_m):
        raise ValueError("weak interval width must be positive")
    domain_half = float(engineering["domain_half_length_m"])
    zeta_left = interval_left_m / domain_half
    zeta_right = interval_right_m / domain_half
    shear_left = model.state(state_index, zeta_left)[:, 3:4]
    shear_right = model.state(state_index, zeta_right)[:, 3:4]

    nodes_np, weights_np = gauss_legendre(int(quadrature_order))
    nodes = torch.as_tensor(
        nodes_np, dtype=interval_left_m.dtype, device=interval_left_m.device
    ).reshape(1, -1)
    weights = torch.as_tensor(
        weights_np, dtype=interval_left_m.dtype, device=interval_left_m.device
    ).reshape(1, -1)
    midpoint = 0.5 * (interval_left_m + interval_right_m)
    half_width = 0.5 * (interval_right_m - interval_left_m)
    quadrature_x = midpoint + half_width * nodes
    quadrature_zeta = (quadrature_x / domain_half).reshape(-1, 1)
    displacement = model.state(state_index, quadrature_zeta)[:, 0:1]
    retention = model.retention(quadrature_zeta, domain_half)
    integrand = (retention * displacement).reshape(interval_left_m.shape[0], -1)
    integral_ru_x = half_width * torch.sum(weights * integrand, dim=1, keepdim=True)

    load_left = float(load_center_m) - 0.5 * float(engineering["load_width_m"])
    load_right = float(load_center_m) + 0.5 * float(engineering["load_width_m"])
    overlap_x = torch.clamp(
        torch.minimum(interval_right_m, torch.full_like(interval_right_m, load_right))
        - torch.maximum(interval_left_m, torch.full_like(interval_left_m, load_left)),
        min=0.0,
    )
    length_scale = scales.length_m
    interval_width_xi = (interval_right_m - interval_left_m) / length_scale
    balance = shear_right - shear_left - 4.0 * (
        integral_ru_x / length_scale - overlap_x / length_scale
    )
    return balance / interval_width_xi


def support_regularization(
    model: InverseMixedPINN,
    *,
    engineering: dict,
    sampling: dict,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    count = int(sampling["regularization_grid_count"])
    outer = float(model.support_field.outer_half_width_m) if isinstance(
        model.support_field, BSplineSupportField
    ) else 15.0
    x_m = torch.linspace(-outer, outer, count, dtype=dtype, device=device).reshape(-1, 1)
    zeta = x_m / float(engineering["domain_half_length_m"])
    support_loss = 1.0 - model.retention(zeta, float(engineering["domain_half_length_m"]))
    dx = (2.0 * outer) / (count - 1)
    first_difference = (support_loss[1:] - support_loss[:-1]) / dx
    return {
        "h1": torch.mean(first_difference.square()),
        "tv": torch.mean(torch.abs(first_difference)),
        "area": torch.mean(support_loss),
    }


def relative_l2(prediction: np.ndarray, reference: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=float)
    reference = np.asarray(reference, dtype=float)
    denominator = float(np.linalg.norm(reference))
    if denominator <= np.finfo(float).tiny:
        raise ValueError("reference norm is zero")
    return float(np.linalg.norm(prediction - reference) / denominator)


def support_metrics(
    x_m: np.ndarray,
    predicted_retention: np.ndarray,
    true_retention: np.ndarray,
    *,
    window_m: tuple[float, float],
    defect_threshold: float,
) -> dict[str, float | int]:
    x_m = np.asarray(x_m, dtype=float)
    predicted_loss = 1.0 - np.asarray(predicted_retention, dtype=float)
    true_loss = 1.0 - np.asarray(true_retention, dtype=float)
    mask = (x_m >= window_m[0]) & (x_m <= window_m[1])
    x = x_m[mask]
    predicted = predicted_loss[mask]
    truth = true_loss[mask]
    true_norm = float(np.linalg.norm(truth))
    loss_rel_l2 = float(np.linalg.norm(predicted - truth) / max(true_norm, np.finfo(float).tiny))
    predicted_region = predicted >= defect_threshold
    true_region = truth >= defect_threshold
    union = int(np.count_nonzero(predicted_region | true_region))
    intersection = int(np.count_nonzero(predicted_region & true_region))
    iou = float(intersection / union) if union else 1.0
    true_area = float(np.trapezoid(truth, x))
    predicted_area = float(np.trapezoid(predicted, x))
    area_rel_error = abs(predicted_area - true_area) / max(true_area, np.finfo(float).tiny)
    true_centroid = float(np.trapezoid(x * truth, x) / max(true_area, np.finfo(float).tiny))
    predicted_centroid = float(
        np.trapezoid(x * predicted, x) / max(predicted_area, np.finfo(float).tiny)
    )

    def component_count(region: np.ndarray) -> int:
        padded = np.pad(region.astype(np.int8), (1, 1))
        return int(np.count_nonzero(np.diff(padded) == 1))

    return {
        "support_loss_rel_l2": loss_rel_l2,
        "defect_region_iou": iou,
        "integrated_loss_true_m": true_area,
        "integrated_loss_predicted_m": predicted_area,
        "integrated_loss_relative_error": float(area_rel_error),
        "true_loss_centroid_m": true_centroid,
        "predicted_loss_centroid_m": predicted_centroid,
        "centroid_absolute_error_m": abs(predicted_centroid - true_centroid),
        "peak_severity_absolute_error": float(abs(np.max(predicted) - np.max(truth))),
        "true_component_count": component_count(true_region),
        "predicted_component_count": component_count(predicted_region),
    }
