"""P2 quality-only must not make an already-trained Dino Reader drift."""
import unittest

from mocap.quality_only import configure_quality_only


class DummyParameter:
    def __init__(self):
        self.requires_grad = True


class DummyLayer:
    def __init__(self, params):
        self.params = params

    def requires_grad_(self, enabled):
        for param in self.params:
            param.requires_grad = bool(enabled)


class DummyReader:
    def __init__(self, quality_head=True):
        self.quality_head = quality_head
        self._params = {
            "backbone.weights": DummyParameter(),
            "neck.weight": DummyParameter(),
            "up1.weight": DummyParameter(),
            "up2.weight": DummyParameter(),
            "out.weight": DummyParameter(),
            "quality_out.weight": DummyParameter(),
            "quality_out.bias": DummyParameter(),
        }
        self.quality_out = DummyLayer([
            self._params["quality_out.weight"],
            self._params["quality_out.bias"],
        ])

    def requires_grad_(self, enabled):
        for p in self._params.values():
            p.requires_grad = bool(enabled)

    def named_parameters(self):
        return iter(self._params.items())


class QualityOnlyTests(unittest.TestCase):
    def test_frozen_pretrained_reader_and_trainable_new_quality_head(self):
        model = DummyReader()
        trainable = configure_quality_only(model)
        self.assertEqual(trainable, ["quality_out.weight", "quality_out.bias"])
        for n, param in model.named_parameters():
            self.assertEqual(param.requires_grad, n.startswith("quality_out."))

    def test_refuses_reader_without_quality_head(self):
        model = DummyReader(quality_head=False)
        with self.assertRaisesRegex(ValueError, "quality_head"):
            configure_quality_only(model)

    def test_torch_optimizer_updates_quality_head_only_when_available(self):
        try:
            import torch
        except ImportError:
            self.skipTest("CPU torch tested in dedicated checkpoint CI job")
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.quality_head = True
                self.backbone = torch.nn.Linear(3, 5)
                self.out = torch.nn.Linear(5, 2)
                self.quality_out = torch.nn.Linear(5, 2)
        model = Model()
        old_backbone = model.backbone.weight.detach().clone()
        old_heatmap = model.out.weight.detach().clone()
        old_quality = model.quality_out.weight.detach().clone()
        configure_quality_only(model)
        opt = torch.optim.AdamW(model.quality_out.parameters(), lr=.05)
        for i in range(4):
            # Freeze model features exactly as train.py --quality-only does.
            model.eval()
            x = torch.ones((7, 3))
            features = model.backbone(x)
            scores = model.quality_out(features)
            loss = ((scores - 2)**2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.assertTrue(torch.equal(model.backbone.weight, old_backbone))
        self.assertTrue(torch.equal(model.out.weight, old_heatmap))
        self.assertFalse(torch.equal(model.quality_out.weight, old_quality))


if __name__ == "__main__":
    unittest.main()
