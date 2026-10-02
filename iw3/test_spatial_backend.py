import unittest
from unittest.mock import patch

import torch

from iw3.spatial_backend import (
    SpatialStereoBackend,
    StereoSettings,
    repair_cpfs_cracks,
    resolve_stereo_geometry,
)


class SpatialBackendGeometryTest(unittest.TestCase):
    def test_learned_mlbw_uses_supported_convergence_and_post_offset(self):
        geometry = resolve_stereo_geometry(
            "mlbw_l2_post",
            width=1536,
            settings=StereoSettings(strength=10.0, parallax_offset_percent=3.75),
        )

        self.assertEqual(geometry.render_convergence, 0.0)
        self.assertAlmostEqual(geometry.left_post_offset_pixels, 57.6)
        self.assertAlmostEqual(geometry.right_post_offset_pixels, -57.6)

    def test_guard_band_preserves_original_pixel_geometry(self):
        geometry = resolve_stereo_geometry(
            "mlbw_l4_post",
            width=1856,
            geometry_width=1536,
            settings=StereoSettings(strength=10.0, parallax_offset_percent=3.75),
        )

        self.assertAlmostEqual(geometry.render_strength, 10.0 * 1536 / 1856)
        self.assertAlmostEqual(geometry.left_post_offset_pixels, 57.6)
        self.assertAlmostEqual(geometry.right_post_offset_pixels, -57.6)

    def test_geometric_renderer_receives_equivalent_native_convergence(self):
        geometry = resolve_stereo_geometry(
            "cpfs_crackfix",
            width=1536,
            settings=StereoSettings(strength=10.0, parallax_offset_percent=3.75),
        )

        self.assertAlmostEqual(geometry.render_convergence, -0.75)
        self.assertEqual(geometry.left_post_offset_pixels, 0.0)
        self.assertEqual(geometry.right_post_offset_pixels, 0.0)


class SpatialBackendModelTest(unittest.TestCase):
    @patch("iw3.spatial_backend.create_stereo_model")
    def test_mlbw_loads_weights_for_requested_strength(self, create_model):
        model = object()
        create_model.return_value = model

        backend = SpatialStereoBackend("mlbw_l2_post", strength=10.0)

        self.assertIs(backend.model, model)
        create_model.assert_called_once_with("mlbw_l2", divergence=10.0, device_id=0)


class CpfsCrackRepairTest(unittest.TestCase):
    @staticmethod
    def inputs():
        alpha = torch.zeros((1, 1, 15, 15), dtype=torch.float32)
        alpha[:, :, 1:14, 1:14] = 1
        color = torch.full((1, 3, 15, 15), 0.6, dtype=torch.float32)
        return color * alpha, alpha, torch.full_like(alpha, 0.5)

    def test_repairs_enclosed_depth_continuous_partial_alpha(self):
        premultiplied, alpha, depth = self.inputs()
        alpha[:, :, 7, 7] = 0.5
        premultiplied[:, :, 7, 7] = 0.3

        repaired_rgb, repaired_alpha, cracks = repair_cpfs_cracks(premultiplied, alpha, depth, strength=10)

        self.assertTrue(cracks[:, :, 7, 7].item())
        self.assertEqual(repaired_alpha[:, :, 7, 7].item(), 1)
        self.assertTrue(torch.allclose(repaired_rgb[:, :, 7, 7], torch.full((1, 3), 0.6)))

    def test_preserves_boundary_holes_and_depth_discontinuities(self):
        cases = ((3, 7, 0.5, 0.5), (7, 7, 0.0, 0.5), (7, 7, 0.5, 0.6))
        for y, x, value, center_depth in cases:
            with self.subTest(y=y, x=x, value=value, depth=center_depth):
                premultiplied, alpha, depth = self.inputs()
                alpha[:, :, y, x] = value
                premultiplied[:, :, y, x] = 0.6 * value
                depth[:, :, y, x] = center_depth

                _, repaired_alpha, cracks = repair_cpfs_cracks(premultiplied, alpha, depth, strength=1024)

                self.assertFalse(cracks[:, :, y, x].item())
                self.assertEqual(repaired_alpha[:, :, y, x].item(), value)


if __name__ == "__main__":
    unittest.main()
