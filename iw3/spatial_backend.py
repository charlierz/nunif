"""Stable frame-level interface for spatial-video stereo rendering."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .backward_warp import apply_divergence_monobw, apply_divergence_nn_LR
from .forward_warp import apply_divergence_forward_warp, depth_order_bilinear_forward_splat
from .mapper import get_mapper
from .models.monobw import MonoBW
from .stereo_model_factory import create_stereo_model

MLBW_RENDERERS = {"mlbw_l2_post", "mlbw_l4_post"}
SUPPORTED_RENDERERS = {
    "cpfs_crackfix",
    "monobw",
    "mlbw_l2_post",
    "mlbw_l4_post",
    "forward_fill",
    "monotonic_math",
}
CPFS_SCALE = 4
CPFS_CRACK_INTERIOR_RADIUS = 4
CPFS_CRACK_DEPTH_THRESHOLD_PIXELS = 2.0
CPFS_CRACK_ALPHA_MAX = 0.99


@dataclass(frozen=True)
class StereoSettings:
    """Renderer-independent horizontal stereo controls."""

    strength: float
    parallax_offset_percent: float
    depth_mapper: str = "none"

    def __post_init__(self):
        if self.strength <= 0:
            raise ValueError("Stereo strength must be positive")


@dataclass(frozen=True)
class ResolvedStereoGeometry:
    """Renderer-native geometry resolved from common controls."""

    render_strength: float
    render_convergence: float
    left_post_offset_pixels: float
    right_post_offset_pixels: float


def resolve_stereo_geometry(
    renderer: str,
    width: int,
    settings: StereoSettings,
    geometry_width: int | None = None,
) -> ResolvedStereoGeometry:
    """Resolve common stereo controls without sending MLBW out of distribution."""
    if renderer not in SUPPORTED_RENDERERS:
        raise ValueError(f"Unsupported spatial renderer: {renderer}")
    if width <= 0 or (geometry_width is not None and geometry_width <= 0):
        raise ValueError("Widths must be positive")

    base_width = geometry_width or width
    render_strength = settings.strength * base_width / width
    offset_pixels = base_width * settings.parallax_offset_percent / 100.0
    if renderer in MLBW_RENDERERS:
        return ResolvedStereoGeometry(
            render_strength=render_strength,
            render_convergence=0.0,
            left_post_offset_pixels=offset_pixels,
            right_post_offset_pixels=-offset_pixels,
        )

    shift_per_convergence = render_strength * 0.01 * width * 0.5
    convergence = -offset_pixels / shift_per_convergence
    return ResolvedStereoGeometry(
        render_strength=render_strength,
        render_convergence=convergence,
        left_post_offset_pixels=0.0,
        right_post_offset_pixels=0.0,
    )


def shift_content(value: torch.Tensor, offset_pixels: float) -> torch.Tensor:
    """Translate content right by a positive number of pixels with transparent borders."""
    batch, _, height, width = value.shape
    y, destination_x = torch.meshgrid(
        torch.linspace(-1, 1, height, device=value.device, dtype=value.dtype),
        torch.arange(width, device=value.device, dtype=value.dtype),
        indexing="ij",
    )
    source_x = destination_x - offset_pixels
    normalized_x = source_x * (2.0 / (width - 1)) - 1.0
    grid = torch.stack((normalized_x, y), dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)
    return F.grid_sample(
        value,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )


def repair_cpfs_cracks(
    premultiplied: torch.Tensor,
    alpha: torch.Tensor,
    projected_depth: torch.Tensor,
    strength: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Seal enclosed partial-alpha cracks without crossing support/depth edges."""
    support = alpha > 0.02
    near_outside = (
        F.max_pool2d(
            (~support).to(alpha.dtype),
            kernel_size=CPFS_CRACK_INTERIOR_RADIUS * 2 + 1,
            stride=1,
            padding=CPFS_CRACK_INTERIOR_RADIUS,
        )
        > 0
    )
    depth_max = F.max_pool2d(projected_depth, kernel_size=3, stride=1, padding=1)
    depth_min = -F.max_pool2d(-projected_depth, kernel_size=3, stride=1, padding=1)
    shift_size = strength * 0.01 * max(alpha.shape[-2:]) * 0.5
    depth_continuous = (depth_max - depth_min) * shift_size < CPFS_CRACK_DEPTH_THRESHOLD_PIXELS
    cracks = (~near_outside) & (alpha > 0.02) & (alpha < CPFS_CRACK_ALPHA_MAX) & depth_continuous
    straight_rgb = torch.where(
        alpha > 1e-6,
        premultiplied / alpha.clamp_min(1e-6),
        torch.zeros_like(premultiplied),
    )
    alpha = torch.where(cracks, torch.ones_like(alpha), alpha)
    return straight_rgb * alpha, alpha, cracks


def cpfs_crackfix_eyes(
    rgb: torch.Tensor,
    alpha: torch.Tensor,
    depth: torch.Tensor,
    strength: float,
    convergence: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Render coverage-preserving forward splats and repair safe interior cracks."""
    binary = (alpha > 0.02).to(alpha.dtype)
    height, width = rgb.shape[-2:]
    high_size = (height, width * CPFS_SCALE)
    rgb_high = F.interpolate(rgb, size=high_size, mode="bilinear", align_corners=True, antialias=True)
    alpha_high = F.interpolate(alpha, size=high_size, mode="bilinear", align_corners=True, antialias=True)
    binary_high = F.interpolate(binary, size=high_size, mode="nearest")
    depth_high = F.interpolate(depth, size=high_size, mode="bilinear", align_corners=True, antialias=True)
    left_rgb, right_rgb, left_alpha, right_alpha = depth_order_bilinear_forward_splat(
        rgb_high, alpha_high, depth_high, strength, convergence
    )
    _, _, left_coverage, right_coverage = depth_order_bilinear_forward_splat(
        torch.zeros_like(rgb_high), binary_high, depth_high, strength, convergence
    )
    left_depth, right_depth, left_depth_alpha, right_depth_alpha = depth_order_bilinear_forward_splat(
        depth_high, binary_high, depth_high, strength, convergence
    )

    eyes = []
    for eye_rgb, eye_alpha, coverage, eye_depth, depth_alpha in zip(
        (left_rgb, right_rgb),
        (left_alpha, right_alpha),
        (left_coverage, right_coverage),
        (left_depth, right_depth),
        (left_depth_alpha, right_depth_alpha),
        strict=True,
    ):
        premultiplied = F.interpolate(eye_rgb * eye_alpha, size=(height, width), mode="area")
        alpha_sum = F.interpolate(eye_alpha, size=(height, width), mode="area")
        coverage = F.interpolate(coverage, size=(height, width), mode="area")
        output_alpha = torch.where(
            coverage > 1e-6,
            alpha_sum / coverage.clamp_min(1e-6),
            torch.zeros_like(alpha_sum),
        ).clamp(0, 1)
        premultiplied = torch.where(
            alpha_sum > 1e-6,
            premultiplied * (output_alpha / alpha_sum.clamp_min(1e-6)),
            torch.zeros_like(premultiplied),
        )
        projected_depth = F.interpolate(eye_depth * depth_alpha, size=(height, width), mode="area")
        depth_alpha = F.interpolate(depth_alpha, size=(height, width), mode="area")
        projected_depth = torch.where(
            depth_alpha > 1e-6,
            projected_depth / depth_alpha.clamp_min(1e-6),
            torch.zeros_like(projected_depth),
        )
        premultiplied, output_alpha, _ = repair_cpfs_cracks(premultiplied, output_alpha, projected_depth, strength)
        eyes.append(torch.cat((premultiplied, output_alpha), dim=1))
    return eyes[0], eyes[1]


class SpatialStereoBackend:
    """Load one IW3-derived renderer and apply it frame by frame."""

    def __init__(self, renderer: str, strength: float, device_id: int = 0):
        if renderer not in SUPPORTED_RENDERERS:
            raise ValueError(f"Unsupported spatial renderer: {renderer}")
        if strength <= 0:
            raise ValueError("Stereo strength must be positive")
        self.renderer = renderer
        self.strength = strength
        self.device_id = device_id
        self.model = self._load_model()

    def _load_model(self):
        if self.renderer == "monobw":
            return create_stereo_model("monobw", divergence=1.0, device_id=self.device_id)
        if self.renderer == "monotonic_math":
            return MonoBW(smooth_kernel=0).to(f"cuda:{self.device_id}").eval()
        if self.renderer.startswith("mlbw_l2"):
            return create_stereo_model("mlbw_l2", divergence=self.strength, device_id=self.device_id)
        if self.renderer.startswith("mlbw_l4"):
            return create_stereo_model("mlbw_l4", divergence=self.strength, device_id=self.device_id)
        return None

    @torch.inference_mode()
    def render(
        self,
        rgb: torch.Tensor,
        alpha: torch.Tensor,
        raw_depth: torch.Tensor,
        settings: StereoSettings,
        geometry_width: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return left/right premultiplied RGBA tensors."""
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError("RGB must have shape Bx3xHxW")
        if alpha.shape != rgb[:, :1].shape:
            raise ValueError("Alpha must have shape Bx1xHxW")
        if raw_depth.ndim != 4 or raw_depth.shape[1] != 1:
            raise ValueError("Depth must have shape Bx1xHxW")

        if settings.strength != self.strength:
            raise ValueError("Stereo strength differs from the loaded renderer model")
        depth = get_mapper(settings.depth_mapper)(raw_depth)
        geometry = resolve_stereo_geometry(self.renderer, rgb.shape[-1], settings, geometry_width)
        rgba = torch.cat((rgb * alpha, alpha), dim=1)
        if self.renderer == "cpfs_crackfix":
            return cpfs_crackfix_eyes(
                rgb,
                alpha,
                depth,
                geometry.render_strength,
                geometry.render_convergence,
            )
        if self.renderer in {"monobw", "monotonic_math"}:
            return apply_divergence_monobw(
                self.model,
                rgba,
                depth,
                divergence=geometry.render_strength,
                convergence=geometry.render_convergence,
                synthetic_view="both",
                preserve_screen_border=False,
            )
        if self.renderer in MLBW_RENDERERS:
            left, right = apply_divergence_nn_LR(
                self.model,
                rgba,
                depth,
                geometry.render_strength,
                geometry.render_convergence,
                steps=None,
                synthetic_view="both",
                preserve_screen_border=False,
                enable_amp=True,
            )
            left = shift_content(left, geometry.left_post_offset_pixels)
            right = shift_content(right, geometry.right_post_offset_pixels)
            return left, right
        if self.renderer == "forward_fill":
            return apply_divergence_forward_warp(
                rgba,
                depth,
                geometry.render_strength,
                geometry.render_convergence,
                method="forward_fill",
                synthetic_view="both",
                width_base=False,
            )
        raise AssertionError(self.renderer)
