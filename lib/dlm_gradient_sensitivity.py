import math

import torch
import torch.nn.functional as F


def midpoint_timesteps(count: int = 10) -> tuple[float, ...]:
    if not isinstance(count, int) or count <= 0:
        raise ValueError("count must be positive")
    return tuple((index + 0.5) / count for index in range(count))


def mask_probability(timestep: float, eps: float = 1e-3) -> float:
    if not math.isfinite(timestep) or not 0 <= timestep <= 1:
        raise ValueError("timestep must be between 0 and 1")
    if not math.isfinite(eps) or not 0 <= eps < 1:
        raise ValueError("eps must be between 0 and 1")
    return (1 - eps) * timestep + eps


def make_masked_state(
    clean_ids: torch.Tensor, timestep: float, mask_id: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, float]:
    if not isinstance(clean_ids, torch.Tensor) or clean_ids.numel() == 0:
        raise ValueError("clean_ids must have a positive sequence length")
    p_mask = mask_probability(timestep)
    generator = torch.Generator(device=clean_ids.device)
    for attempt in range(clean_ids.numel() + 1):
        generator.manual_seed(seed + attempt)
        mask = torch.rand(
            clean_ids.shape, device=clean_ids.device, generator=generator
        ).lt(p_mask)
        if mask.any():
            masked_ids = clean_ids.clone()
            masked_ids[mask] = mask_id
            return masked_ids, mask, p_mask
    raise RuntimeError("deterministic mask resampling failed")


def official_dlm_loss(
    logits: torch.Tensor,
    clean_ids: torch.Tensor,
    mask: torch.Tensor,
    p_mask: float,
) -> torch.Tensor:
    if clean_ids.numel() == 0:
        raise ValueError("clean_ids must have a positive sequence length")
    if logits.ndim == 0 or tuple(logits.shape[:-1]) != tuple(clean_ids.shape):
        raise ValueError("logits and clean_ids shapes do not match")
    if mask.shape != clean_ids.shape or mask.dtype != torch.bool:
        raise ValueError("mask shape and dtype do not match clean_ids")
    if not math.isfinite(p_mask) or not 0 < p_mask <= 1:
        raise ValueError("p_mask must be in (0, 1]")
    if not mask.any():
        raise ValueError("official DLM loss requires at least one masked token")
    token_loss = F.cross_entropy(logits[mask].float(), clean_ids[mask], reduction="sum")
    return token_loss / p_mask / clean_ids.numel()


class TimestepSensitivityAccumulator:
    def __init__(
        self,
        weights: dict[str, torch.Tensor],
        split_size: int = 4,
        expected_timesteps: int = 10,
    ) -> None:
        if not weights:
            raise ValueError("weights must not be empty")
        if not isinstance(split_size, int) or split_size <= 0:
            raise ValueError("split_size must be positive")
        if not isinstance(expected_timesteps, int) or expected_timesteps <= 0:
            raise ValueError("expected_timesteps must be positive")
        self.split_size = split_size
        self.expected_timesteps = expected_timesteps
        self._weights = {
            name: weight.detach().to(device="cpu", dtype=torch.float32).clone()
            for name, weight in weights.items()
        }
        if any(not torch.isfinite(weight).all().item() for weight in self._weights.values()):
            raise ValueError("weights must be finite")
        self._weight_sq = {name: weight.square() for name, weight in self._weights.items()}
        self._sum_sq = {
            split: {name: torch.zeros_like(weight) for name, weight in self._weights.items()}
            for split in ("a", "b")
        }
        self._count = {"a": 0, "b": 0}
        self._mean = {
            group: {name: torch.zeros_like(weight) for name, weight in self._weights.items()}
            for group in ("full", "a", "b")
        }
        self._m2 = {
            group: {name: torch.zeros_like(weight) for name, weight in self._weights.items()}
            for group in ("full", "a", "b")
        }
        self._update_count = 0

    def add_state(self, grads: dict[str, torch.Tensor], split: str) -> None:
        if split not in self._count:
            raise ValueError(f"unknown split: {split}")
        if self._count[split] >= self.split_size:
            raise ValueError(f"split {split} already has split_size states")
        if set(grads) != set(self._weights):
            raise ValueError("gradient keys do not match weights")
        cpu_grads = {}
        for name, grad in grads.items():
            if not isinstance(grad, torch.Tensor) or grad.shape != self._weights[name].shape:
                raise ValueError(f"gradient shape does not match weight: {name}")
            if not torch.isfinite(grad).all().item():
                raise ValueError("gradients must be finite")
            cpu_grads[name] = grad.detach().to(device="cpu", dtype=torch.float32).square()
        for name, grad_sq in cpu_grads.items():
            self._sum_sq[split][name].add_(grad_sq)
        self._count[split] += 1

    def finish_timestep(self) -> None:
        if self._update_count >= self.expected_timesteps:
            raise ValueError("expected timestep updates already complete")
        if any(count != self.split_size for count in self._count.values()):
            raise ValueError("cannot finish an incomplete timestep")
        f_a = {
            name: self._weight_sq[name] * self._sum_sq["a"][name] / self.split_size
            for name in self._weights
        }
        f_b = {
            name: self._weight_sq[name] * self._sum_sq["b"][name] / self.split_size
            for name in self._weights
        }
        values = {
            "a": f_a,
            "b": f_b,
            "full": {name: (f_a[name] + f_b[name]) / 2 for name in self._weights},
        }
        next_count = self._update_count + 1
        for group, group_values in values.items():
            for name, value in group_values.items():
                delta = value - self._mean[group][name]
                self._mean[group][name].add_(delta / next_count)
                self._m2[group][name].add_(delta * (value - self._mean[group][name]))
        for split in self._count:
            self._count[split] = 0
            for value in self._sum_sq[split].values():
                value.zero_()
        self._update_count = next_count

    def finalize(self) -> dict:
        if any(self._count.values()):
            raise ValueError("cannot finalize an incomplete timestep")
        if self._update_count != self.expected_timesteps:
            raise ValueError(
                f"expected {self.expected_timesteps} timestep updates, got {self._update_count}"
            )
        result = {"update_count": self._update_count}
        for group in ("full", "a", "b"):
            result[group] = {
                "mu": {name: value.clone() for name, value in self._mean[group].items()},
                "sigma": {
                    name: torch.sqrt(value / self._update_count)
                    for name, value in self._m2[group].items()
                },
            }
        return result
