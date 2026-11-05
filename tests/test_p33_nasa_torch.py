"""Tiny synthetic wrapper tests. Passing them does not validate the NASA model."""
from pathlib import Path
import sys
import tempfile
import unittest
import types
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from torch import nn
    from lunarrisk.nasa_transfer.contracts import file_sha256
    from lunarrisk.nasa_transfer.torch_adapter import (
        TerrainTransfer, build_paired_models, load_verified_checkpoint, _verify_import_origins,
    )

    class TinyBackbone(nn.Module):
        """Deliberately NOT the NASA architecture; exercises the integration only."""
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.encoder_embeddings = nn.ModuleDict({key: nn.Conv2d(1, 4, 2, 2) for key in ("dtm", "slope")})
            for embedding in self.model.encoder_embeddings.values():
                embedding.image_size = (4, 4)
            self.model.encoder = nn.ModuleList([nn.Linear(4, 4)])
            self.model.encoder_norm = nn.LayerNorm(4)
            self.model.register_tokens = nn.Parameter(torch.randn(1, 1, 4))
            self.out_channels = [4]
            self.merge_method = None
            self.remove_register_tokens = True

        def forward(self, inputs):
            tokens = torch.cat([self.model.encoder_embeddings[key](inputs[key]).flatten(2).transpose(1, 2)
                                for key in ("dtm", "slope")], dim=1)
            return [self.model.encoder_norm(self.model.encoder[0](tokens))]


@unittest.skipIf(torch is None, "requires compatible private torch runtime; NASA itself is not exercised")
class TestNasaTorchWrapper(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.inputs = {key: torch.randn(2, 1, 4, 4) for key in ("dtm", "slope")}

    def test_frozen_encoder_stays_eval_and_no_encoder_gradients(self):
        model = TerrainTransfer(TinyBackbone()).train()
        self.assertFalse(model.backbone.training)
        model(self.inputs).sum().backward()
        self.assertTrue(all(parameter.grad is None for parameter in model.backbone.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in model.adapters.parameters()))
        self.assertTrue(all(parameter.grad is not None for parameter in model.head.parameters()))

    def test_adapter_gradients_flow_through_frozen_encoder(self):
        model = TerrainTransfer(TinyBackbone(), adapt_inputs=True).train()
        loss = nn.functional.cross_entropy(model(self.inputs), torch.tensor([0, 48]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(parameter.grad is None for parameter in model.backbone.parameters()))
        for name, parameter in model.adapters.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0, name)

    def test_identity_adapter_matches_unadapted_backbone(self):
        model = TerrainTransfer(TinyBackbone()).eval()
        expected = model.head(model.backbone(self.inputs)[-1].mean(1))
        torch.testing.assert_close(model(self.inputs), expected, rtol=0, atol=0)

    def test_pair_has_same_heads_and_adapters_but_different_backbone_weights(self):
        def load(backbone):
            with torch.no_grad():
                backbone.model.encoder[0].weight.add_(1)
            return {"kind": "synthetic fixture only"}
        state_before = torch.get_rng_state().clone()
        models, report = build_paired_models(TinyBackbone, load, adapt_inputs=True)
        self.assertTrue(torch.equal(state_before, torch.get_rng_state()))
        self.assertNotEqual(report["kind"], "pretrained")
        for part in ("head", "adapters"):
            for key, value in getattr(models["random"], part).state_dict().items():
                torch.testing.assert_close(value, getattr(models["pretrained"], part).state_dict()[key], rtol=0, atol=0)
        self.assertFalse(torch.equal(models["random"].backbone.model.encoder[0].weight,
                                     models["pretrained"].backbone.model.encoder[0].weight))

    def test_checkpoint_integrity_and_exact_inference_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.pt"
            original = TinyBackbone()
            torch.save({"model": original.model.state_dict()}, path)
            loaded = TinyBackbone()
            report = load_verified_checkpoint(loaded, path, expected_bytes=path.stat().st_size,
                                               expected_sha256=file_sha256(path))
            self.assertEqual(report["loaded_inference_fraction"], 1)
            for key, value in original.model.state_dict().items():
                torch.testing.assert_close(value, loaded.model.state_dict()[key], rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, "SHA256"):
                load_verified_checkpoint(loaded, path, expected_bytes=path.stat().st_size,
                                         expected_sha256="0" * 64)

    def test_missing_or_nonfinite_checkpoint_tensor_is_not_success(self):
        for kind in ("missing", "nan"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                backbone = TinyBackbone()
                state = backbone.model.state_dict()
                if kind == "missing":
                    del state["encoder.0.weight"]
                else:
                    state["encoder.0.weight"][0, 0] = float("nan")
                path = Path(directory) / "synthetic.pt"
                torch.save(state, path)
                with self.assertRaises(ValueError):
                    load_verified_checkpoint(TinyBackbone(), path, expected_bytes=path.stat().st_size,
                                             expected_sha256=file_sha256(path))

    def test_noncanonical_patch_divisible_shape_is_rejected(self):
        model = TerrainTransfer(TinyBackbone())
        inputs = dict(self.inputs, dtm=torch.randn(2, 1, 8, 8))
        with self.assertRaisesRegex(ValueError, "canonical"):
            model(inputs)

    def test_preloaded_upstream_module_outside_pin_is_rejected(self):
        module = types.ModuleType("ni_lfm.fake_audit_test")
        module.__file__ = "/not/the/audited/source.py"
        with patch.dict(sys.modules, {module.__name__: module}):
            with self.assertRaisesRegex(RuntimeError, "unpinned"):
                _verify_import_origins(Path("/audited"))

    def test_masked_zero_merge_or_register_double_strip_rejected(self):
        for kind in ("merge", "register"):
            backbone = TinyBackbone()
            if kind == "merge":
                backbone.merge_method = "masked_group_mean"
            else:
                backbone.remove_register_tokens = False
            with self.assertRaises(ValueError):
                TerrainTransfer(backbone)


if __name__ == "__main__":
    unittest.main()
