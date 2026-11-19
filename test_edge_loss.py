"""Small CPU checks; no pretrained weights or image downloads needed."""
import unittest

import torch
from torch import nn

from edge_loss import SobelEdgeLoss
from pilot import transfer, build_model
from main import get_style_model_and_losses


class EdgeLossTests(unittest.TestCase):
    def test_identical_images_and_constant_padding(self):
        target = torch.ones(1, 3, 12, 16)
        loss = SobelEdgeLoss(target)
        self.assertEqual(loss(target).item(), 0)
        self.assertEqual(loss.edges(target).abs().max().item(), 0)

    def test_changed_edges_backpropagate_only_to_output(self):
        target = torch.zeros(1, 3, 12, 16, requires_grad=True)
        image = torch.zeros_like(target).detach()
        image[:, :, :, 8:] = 1
        image.requires_grad_(True)
        value = SobelEdgeLoss(target)(image)
        value.backward()
        self.assertGreater(value.item(), 0)
        self.assertGreater(image.grad.abs().sum().item(), 0)
        self.assertIsNone(target.grad)

    def test_cpu_optimization_smoke(self):
        torch.manual_seed(0)
        layers = []
        for _ in range(5):
            layers.extend([nn.Conv2d(3, 3, 3, padding=1), nn.ReLU()])
        cnn = nn.Sequential(*layers).eval().requires_grad_(False)
        content, style = torch.rand(1, 3, 16, 20), torch.rand(1, 3, 18, 16)
        for weight in [0, 100]:
            output, history, metrics = transfer(cnn, content, style, 2, 1e3, 1, weight)
            self.assertEqual(output.shape, content.shape)
            self.assertEqual(len(history), 2)
            self.assertTrue(torch.isfinite(output).all())
            self.assertTrue(0 <= output.min() <= output.max() <= 1)
            self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in metrics.values()))
            self.assertTrue(all(p.grad is None for p in cnn.parameters()))

    def test_baseline_losses_match_original_builder(self):
        torch.manual_seed(1)
        cnn = nn.Sequential(*[nn.Conv2d(3, 3, 3, padding=1) for _ in range(5)]).requires_grad_(False)
        content, style = torch.rand(1, 3, 16, 20), torch.rand(1, 3, 16, 20)
        mean, std = torch.zeros(3), torch.ones(3)
        old, old_s, old_c = get_style_model_and_losses(cnn, mean, std, style, content)
        new, new_s, new_c = build_model(cnn, mean, std, content, style)
        probe = torch.rand_like(content)
        old(probe)
        new(probe)
        torch.testing.assert_close(sum(x.loss for x in old_s), sum(x.loss for x in new_s))
        torch.testing.assert_close(sum(x.loss for x in old_c), sum(x.loss for x in new_c))


if __name__ == "__main__":
    unittest.main()
