"""Mixed-state network representations for the registered Phase 6C screen."""

from __future__ import annotations

import math

import torch
from torch import nn


def hard_boundary_states(raw: torch.Tensor, zeta: torch.Tensor) -> torch.Tensor:
    boundary_factor = 1.0 - zeta.square()
    displacement = boundary_factor.square() * raw[:, 0:1]
    rotation = boundary_factor * raw[:, 1:2]
    return torch.cat((displacement, rotation, raw[:, 2:3], raw[:, 3:4]), dim=1)


class SineLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        omega_0: float,
        is_first: bool,
    ) -> None:
        super().__init__()
        self.omega_0 = float(omega_0)
        self.linear = nn.Linear(in_features, out_features)
        with torch.no_grad():
            if is_first:
                bound = 1.0 / in_features
            else:
                bound = math.sqrt(6.0 / in_features) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(values))


class SirenMixedPINN(nn.Module):
    def __init__(
        self,
        hidden_layers: int,
        width: int,
        *,
        first_omega_0: float,
        hidden_omega_0: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_features = 1
        for index in range(hidden_layers):
            omega = first_omega_0 if index == 0 else hidden_omega_0
            layers.append(
                SineLayer(
                    in_features,
                    width,
                    omega_0=omega,
                    is_first=index == 0,
                )
            )
            in_features = width
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(width, 4)
        bound = math.sqrt(6.0 / width) / float(hidden_omega_0)
        with torch.no_grad():
            self.output.weight.uniform_(-bound, bound)
            self.output.bias.uniform_(-bound, bound)

    def forward(self, zeta: torch.Tensor) -> torch.Tensor:
        return hard_boundary_states(self.output(self.hidden(zeta)), zeta)


class FourierFeatures(nn.Module):
    def __init__(self, frequencies: list[float]) -> None:
        super().__init__()
        if not frequencies or min(frequencies) <= 0.0:
            raise ValueError("Fourier frequencies must be positive")
        self.register_buffer(
            "frequencies", torch.tensor(frequencies, dtype=torch.get_default_dtype())
        )

    @property
    def output_size(self) -> int:
        return 1 + 2 * int(self.frequencies.numel())

    def forward(self, zeta: torch.Tensor) -> torch.Tensor:
        angles = zeta * self.frequencies.to(dtype=zeta.dtype).reshape(1, -1)
        scale = 1.0 / math.sqrt(float(self.frequencies.numel()))
        return torch.cat((zeta, scale * torch.sin(angles), scale * torch.cos(angles)), dim=1)


class TanhFeatureMixedPINN(nn.Module):
    def __init__(self, hidden_layers: int, width: int, frequencies: list[float]) -> None:
        super().__init__()
        self.features = FourierFeatures(frequencies)
        modules: list[nn.Module] = []
        in_features = self.features.output_size
        for _ in range(hidden_layers):
            linear = nn.Linear(in_features, width)
            nn.init.xavier_normal_(linear.weight)
            nn.init.zeros_(linear.bias)
            modules.extend((linear, nn.Tanh()))
            in_features = width
        output = nn.Linear(in_features, 4)
        nn.init.xavier_normal_(output.weight)
        nn.init.zeros_(output.bias)
        modules.append(output)
        self.network = nn.Sequential(*modules)

    def forward(self, zeta: torch.Tensor) -> torch.Tensor:
        return hard_boundary_states(self.network(self.features(zeta)), zeta)


class FullWaveActivation(nn.Module):
    """Layerwise trainable implementation of the author's sine–cosine form."""

    def __init__(
        self,
        *,
        sine_coefficient: float,
        cosine_coefficient: float,
        frequency: float,
        phase: float,
    ) -> None:
        super().__init__()
        if frequency <= 0.0:
            raise ValueError("initial Full-Wave frequency must be positive")
        self.sine_coefficient = nn.Parameter(torch.tensor(float(sine_coefficient)))
        self.cosine_coefficient = nn.Parameter(torch.tensor(float(cosine_coefficient)))
        self.log_frequency = nn.Parameter(torch.tensor(math.log(float(frequency))))
        self.phase = nn.Parameter(torch.tensor(float(phase)))

    @property
    def frequency(self) -> torch.Tensor:
        return torch.exp(self.log_frequency)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        argument = self.frequency.to(values.dtype) * values + self.phase.to(values.dtype)
        return self.sine_coefficient.to(values.dtype) * torch.sin(argument) + self.cosine_coefficient.to(values.dtype) * torch.cos(argument)


class GlobalPhysicsFeatures(nn.Module):
    def __init__(self, frequency: float) -> None:
        super().__init__()
        self.frequency = float(frequency)
        if self.frequency <= 0.0:
            raise ValueError("physics frequency must be positive")
        self.output_size = 3

    def forward(self, zeta: torch.Tensor) -> torch.Tensor:
        angle = self.frequency * zeta
        return torch.cat((zeta, torch.sin(angle), torch.cos(angle)), dim=1)


class LocalizedPhysicsFeatures(nn.Module):
    def __init__(
        self,
        *,
        anchors_zeta: list[float],
        frequency: float,
        decay: float,
        smooth_distance_epsilon: float,
    ) -> None:
        super().__init__()
        if not anchors_zeta or min(frequency, decay, smooth_distance_epsilon) <= 0.0:
            raise ValueError("localized feature settings must be positive and non-empty")
        self.register_buffer(
            "anchors_zeta", torch.tensor(anchors_zeta, dtype=torch.get_default_dtype())
        )
        self.frequency = float(frequency)
        self.decay = float(decay)
        self.smooth_distance_epsilon = float(smooth_distance_epsilon)
        self.output_size = 1 + 2 * len(anchors_zeta)

    def forward(self, zeta: torch.Tensor) -> torch.Tensor:
        offsets = zeta - self.anchors_zeta.to(dtype=zeta.dtype).reshape(1, -1)
        distance = torch.sqrt(offsets.square() + self.smooth_distance_epsilon**2)
        envelope = torch.exp(-self.decay * distance)
        angle = self.frequency * distance
        return torch.cat(
            (zeta, envelope * torch.sin(angle), envelope * torch.cos(angle)), dim=1
        )


class FullWaveMixedPINN(nn.Module):
    def __init__(
        self,
        hidden_layers: int,
        width: int,
        *,
        feature_module: nn.Module | None,
        feature_size: int,
        activation_settings: dict,
    ) -> None:
        super().__init__()
        self.features = feature_module
        self.linears = nn.ModuleList()
        self.activations = nn.ModuleList()
        in_features = feature_size
        for _ in range(hidden_layers):
            linear = nn.Linear(in_features, width)
            nn.init.xavier_normal_(linear.weight)
            nn.init.zeros_(linear.bias)
            self.linears.append(linear)
            self.activations.append(
                FullWaveActivation(
                    sine_coefficient=float(
                        activation_settings["initial_sine_coefficient"]
                    ),
                    cosine_coefficient=float(
                        activation_settings["initial_cosine_coefficient"]
                    ),
                    frequency=float(activation_settings["initial_frequency"]),
                    phase=float(activation_settings["initial_phase"]),
                )
            )
            in_features = width
        self.output = nn.Linear(in_features, 4)
        nn.init.xavier_normal_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, zeta: torch.Tensor) -> torch.Tensor:
        values = zeta if self.features is None else self.features(zeta)
        for linear, activation in zip(self.linears, self.activations, strict=True):
            values = activation(linear(values))
        return hard_boundary_states(self.output(values), zeta)

    def frequency_penalty(self) -> torch.Tensor:
        return torch.mean(
            torch.stack([activation.log_frequency.square() for activation in self.activations])
        )


def build_representation_model(
    candidate_id: str,
    *,
    hidden_layers: int,
    width: int,
    representation: dict,
    engineering: dict,
) -> nn.Module:
    if candidate_id == "SIREN":
        settings = representation["siren"]
        return SirenMixedPINN(
            hidden_layers,
            width,
            first_omega_0=float(settings["first_omega_0"]),
            hidden_omega_0=float(settings["hidden_omega_0"]),
        )
    if candidate_id == "FOURIER":
        return TanhFeatureMixedPINN(
            hidden_layers,
            width,
            list(representation["fourier"]["frequencies"]),
        )
    fullwave = representation["fullwave"]
    if candidate_id == "FW_ORIGINAL":
        return FullWaveMixedPINN(
            hidden_layers,
            width,
            feature_module=None,
            feature_size=1,
            activation_settings=fullwave,
        )
    physics_frequency = float(representation["physics_frequency"])
    if candidate_id == "FW_PHYSICS":
        features = GlobalPhysicsFeatures(physics_frequency)
        return FullWaveMixedPINN(
            hidden_layers,
            width,
            feature_module=features,
            feature_size=features.output_size,
            activation_settings=fullwave,
        )
    if candidate_id == "FW_LOCALIZED":
        domain_half = float(engineering["domain_half_length_m"])
        center = float(engineering["load_center_m"]) / domain_half
        half_width = 0.5 * float(engineering["load_width_m"]) / domain_half
        settings = representation["localized"]
        features = LocalizedPhysicsFeatures(
            anchors_zeta=[center - half_width, center, center + half_width],
            frequency=physics_frequency * float(settings["frequency_multiplier"]),
            decay=physics_frequency * float(settings["decay_multiplier"]),
            smooth_distance_epsilon=float(settings["smooth_distance_epsilon_zeta"]),
        )
        return FullWaveMixedPINN(
            hidden_layers,
            width,
            feature_module=features,
            feature_size=features.output_size,
            activation_settings=fullwave,
        )
    raise ValueError(f"unknown representation candidate: {candidate_id}")

