import unittest

import torch

from iw3.forward_warp import depth_order_bilinear_forward_splat


class ForwardWarpCoverageTest(unittest.TestCase):
    def test_identity_preserves_rgb_and_alpha(self):
        rgb = torch.tensor(
            [[[[0.1, 0.2, 0.3, 0.4]],
              [[0.5, 0.6, 0.7, 0.8]],
              [[0.9, 0.8, 0.7, 0.6]]]],
            dtype=torch.float32)
        alpha = torch.tensor([[[[0.25, 0.5, 0.75, 1.0]]]], dtype=torch.float32)
        depth = torch.zeros((1, 1, 1, 4), dtype=torch.float32)

        left_rgb, right_rgb, left_alpha, right_alpha = depth_order_bilinear_forward_splat(
            rgb, alpha, depth, divergence=0, convergence=0)

        torch.testing.assert_close(left_rgb, rgb)
        torch.testing.assert_close(right_rgb, rgb)
        torch.testing.assert_close(left_alpha, alpha)
        torch.testing.assert_close(right_alpha, alpha)

    def test_preserves_partial_coverage_during_expansion(self):
        width = 16
        rgb = torch.ones((1, 3, 1, width), dtype=torch.float32)
        alpha = torch.ones((1, 1, 1, width), dtype=torch.float32)
        depth = torch.linspace(0, 1, width).view(1, 1, 1, width)

        _, _, left_alpha, right_alpha = depth_order_bilinear_forward_splat(
            rgb, alpha, depth, divergence=50, convergence=0.5)

        self.assertTrue(((left_alpha > 0) & (left_alpha < 0.99)).any())
        self.assertTrue(((right_alpha > 0) & (right_alpha < 0.99)).any())

    def test_does_not_fill_disocclusion_holes(self):
        width = 16
        rgb = torch.ones((1, 3, 1, width), dtype=torch.float32)
        alpha = torch.ones((1, 1, 1, width), dtype=torch.float32)
        depth = torch.zeros((1, 1, 1, width), dtype=torch.float32)
        depth[:, :, :, width // 2:] = 1

        _, _, left_alpha, right_alpha = depth_order_bilinear_forward_splat(
            rgb, alpha, depth, divergence=50, convergence=0.5)

        self.assertTrue((left_alpha == 0).any())
        self.assertTrue((right_alpha == 0).any())


if __name__ == "__main__":
    unittest.main()
