"""P32 same-context head contracts; synthetic CPU tests, no model downloads."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from lunarrisk.lunarjev.heads import build_head, masked_probabilities


METHODS = ("B0", "B1", "B2", "J")


@unittest.skipIf(torch is None, "requires the existing private torch runtime")
class TestP32Heads(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(42)
        self.text = {
            "geology": torch.randn(49, 12),
            "age": torch.randn(5, 12),
        }
        self.tasks = {task: torch.randn(12) for task in self.text}
        self.tokens = torch.randn(3, 16, 256)

    def head(self, method, token_width=256):
        return build_head(method, self.text, self.tasks, token_width=token_width)

    def test_task_output_widths_and_forward_all(self):
        for method in METHODS:
            with self.subTest(method=method), torch.no_grad():
                head = self.head(method).eval()
                results = head.forward_all(self.tokens)
                self.assertEqual(set(results), {"geology", "age"})
                for task, classes in (("geology", 49), ("age", 5)):
                    self.assertEqual(results[task].shape, (3, classes))
                    self.assertTrue(torch.isfinite(results[task]).all())
                    torch.testing.assert_close(results[task], head(self.tokens, task))

    def test_permutation_and_subset_mask_identity(self):
        for method in METHODS:
            head = self.head(method).eval()
            for task, classes in (("geology", 49), ("age", 5)):
                with self.subTest(method=method, task=task), torch.no_grad():
                    full = head(self.tokens, task)
                    permutation = torch.randperm(classes)
                    permuted = head(self.tokens, task, permutation)
                    torch.testing.assert_close(permuted[:, permutation.argsort()], full, rtol=1e-5, atol=1e-6)
                    for values in ([classes - 1], [classes - 1, 0, 2]):
                        ids = torch.tensor(values, dtype=torch.int32)
                        selected = head(self.tokens, task, ids)
                        torch.testing.assert_close(selected, full[:, ids.long()], rtol=1e-5, atol=1e-6)
                        mask = torch.zeros(classes, dtype=torch.bool)
                        mask[ids.long()] = True
                        expected = masked_probabilities(full, mask)[:, ids.long()]
                        torch.testing.assert_close(selected.softmax(-1), expected, rtol=1e-5, atol=1e-6)

    def test_invalid_candidate_ids(self):
        invalid = (
            [], torch.tensor([], dtype=torch.long), [0, 0], [-1], [49],
            [0.0, 1.0], [False, True], [0, True], "01", 1,
            torch.tensor([0.0, 1.0]), torch.tensor([False, True]),
            torch.tensor([[0, 1]]), torch.tensor(0), [[0], [1]],
            [2**100],
        )
        for method in METHODS:
            head = self.head(method)
            for ids in invalid:
                with self.subTest(method=method, ids=ids):
                    with self.assertRaises(ValueError):
                        head(self.tokens, "geology", ids)
            with self.assertRaises(ValueError):
                head(self.tokens, "age", [5])

    def test_candidate_integer_sequences_preserve_order(self):
        head = self.head("J").eval()
        with torch.no_grad():
            expected = head(self.tokens, "age")[:, [4, 0, 2]]
            for ids in ([4, 0, 2], (4, 0, 2), torch.tensor([4, 0, 2], dtype=torch.int16)):
                with self.subTest(ids=ids):
                    torch.testing.assert_close(head(self.tokens, "age", ids), expected, rtol=1e-5, atol=1e-6)

    def test_finite_training_gradients_for_both_tasks(self):
        for method in METHODS:
            with self.subTest(method=method):
                head = self.head(method).train()
                tokens = self.tokens.clone().requires_grad_(True)
                results = head.forward_all(tokens)
                loss = sum(torch.nn.functional.cross_entropy(results[task], labels) for task, labels in (
                    ("geology", torch.tensor([0, 1, 48])),
                    ("age", torch.tensor([4, 1, 0])),
                )) / 2
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertIsNotNone(tokens.grad)
                self.assertTrue(torch.isfinite(tokens.grad).all())
                self.assertGreater(tokens.grad.abs().sum().item(), 0)
                for name, parameter in head.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                torch.optim.AdamW(head.parameters(), lr=1e-3).step()
                for output in head.forward_all(self.tokens).values():
                    self.assertTrue(torch.isfinite(output).all())

    def test_public_features_are_owned_frozen_persistent_buffers(self):
        for method in METHODS:
            with self.subTest(method=method):
                text = {key: value.clone().requires_grad_(True) for key, value in self.text.items()}
                tasks = {key: value.clone().requires_grad_(True) for key, value in self.tasks.items()}
                head = build_head(method, text, tasks)
                original = head.text_geology.clone()
                with torch.no_grad():
                    text["geology"].add_(100)
                    tasks["geology"].add_(100)
                torch.testing.assert_close(head.text_geology, original)
                for task in self.text:
                    for name in (f"text_{task}", f"task_{task}"):
                        self.assertIn(name, dict(head.named_buffers()))
                        self.assertIn(name, head.state_dict())
                        self.assertNotIn(name, dict(head.named_parameters()))
                        self.assertFalse(getattr(head, name).requires_grad)
                sum(value.sum() for value in head.forward_all(self.tokens).values()).backward()
                for value in (*text.values(), *tasks.values()):
                    self.assertIsNone(value.grad)

    def test_state_dict_restores_logits_and_descriptor_tables(self):
        for method in METHODS:
            with self.subTest(method=method), torch.no_grad():
                head = self.head(method).eval()
                original = head.forward_all(self.tokens)
                restored = self.head(method).eval()
                restored.load_state_dict(head.state_dict())
                for task, logits in restored.forward_all(self.tokens).items():
                    torch.testing.assert_close(logits, original[task], rtol=0, atol=0)
                torch.testing.assert_close(restored.text_geology, head.text_geology, rtol=0, atol=0)

    def test_semantic_heads_have_no_learned_class_id_shortcut(self):
        for method in ("B2", "J"):
            with self.subTest(method=method):
                head = self.head(method).train()
                self.assertFalse(any(isinstance(module, torch.nn.Embedding) for module in head.modules()))
                for name, parameter in head.named_parameters():
                    self.assertNotIn(parameter.shape[0], (49, 5), name)
                with torch.no_grad():
                    head.text_geology[1].copy_(head.text_geology[0])
                # Contradictory target IDs cannot split identical descriptors:
                # all learned paths use shared weights rather than class IDs.
                optimizer = torch.optim.SGD(head.parameters(), lr=0.01)
                for _ in range(2):
                    logits = head(self.tokens, "geology")
                    torch.testing.assert_close(logits[:, 0], logits[:, 1], rtol=1e-5, atol=1e-6)
                    optimizer.zero_grad()
                    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 0])).backward()
                    optimizer.step()

    def test_semantic_logits_follow_descriptor_rows_and_task_text(self):
        for method in ("B2", "J"):
            with self.subTest(method=method), torch.no_grad():
                head = self.head(method).eval()
                original = head(self.tokens, "geology")
                permutation = torch.randperm(49)
                head.text_geology.copy_(head.text_geology[permutation])
                actual = head(self.tokens, "geology")
                torch.testing.assert_close(actual, original[:, permutation], rtol=1e-5, atol=1e-6)
                head.task_geology.add_(torch.randn_like(head.task_geology))
                self.assertFalse(torch.allclose(head(self.tokens, "geology"), actual))

    def test_ordinary_and_id_controls_ignore_descriptor_values(self):
        for method in ("B0", "B1"):
            with self.subTest(method=method), torch.no_grad():
                head = self.head(method).eval()
                original = head.forward_all(self.tokens)
                for task in self.text:
                    getattr(head, f"text_{task}").normal_(mean=100)
                    getattr(head, f"task_{task}").normal_(mean=100)
                for task, logits in head.forward_all(self.tokens).items():
                    torch.testing.assert_close(logits, original[task], rtol=0, atol=0)
        self.assertTrue(any(isinstance(module, torch.nn.Embedding) for module in self.head("B1").modules()))

    def test_pooled_controls_ignore_spatial_token_order(self):
        permutation = torch.randperm(16)
        for method in ("B0", "B1", "B2"):
            with self.subTest(method=method), torch.no_grad():
                head = self.head(method).eval()
                original = head.forward_all(self.tokens)
                for task, logits in head.forward_all(self.tokens[:, permutation]).items():
                    torch.testing.assert_close(logits, original[task], rtol=1e-5, atol=1e-6)

    def test_j_alone_uses_fixed_2d_spatial_positions(self):
        with torch.no_grad():
            head = self.head("J").eval()
            positions = head.spatial_positions
            self.assertEqual(positions.shape, (16, 128))
            self.assertEqual(positions.unique(dim=0).shape[0], 16)
            self.assertFalse(positions.requires_grad)
            self.assertNotIn("spatial_positions", dict(head.named_parameters()))
            # Horizontal/vertical steps occupy separate sinusoidal channels.
            torch.testing.assert_close(positions[0, :64], positions[1, :64])
            torch.testing.assert_close(positions[0, 64:], positions[4, 64:])
            original = head(self.tokens, "geology")
            reversed_tokens = head(self.tokens.flip(1), "geology")
            self.assertFalse(torch.allclose(original, reversed_tokens))
            saved_positions = positions.clone()
            head.spatial_positions.zero_()
            without_positions = head(self.tokens, "geology")
            torch.testing.assert_close(without_positions, head(self.tokens.flip(1), "geology"), rtol=1e-5, atol=1e-6)
            head.spatial_positions.copy_(saved_positions)
        for method in ("B0", "B1", "B2"):
            self.assertFalse(hasattr(self.head(method), "spatial_positions"))

    def test_query_architecture_and_one_context_preparation_for_both_tasks(self):
        for method in ("B1", "B2", "J"):
            with self.subTest(method=method):
                head = self.head(method)
                self.assertEqual(len(head.blocks), 2)
                for block in head.blocks:
                    self.assertEqual(block.attention.embed_dim, 128)
                    self.assertEqual(block.attention.num_heads, 4)
                    self.assertEqual(block.attention.dropout, 0)
                    self.assertEqual(block.feedforward.layers[0].out_features, 256)
                seen = []
                handle = head.visual_projection.register_forward_pre_hook(lambda module, args: seen.append(args[0].detach()))
                try:
                    head.forward_all(self.tokens)
                finally:
                    handle.remove()
                self.assertEqual(len(seen), 1)
                if method == "J":
                    torch.testing.assert_close(seen[0], self.tokens)
                else:
                    torch.testing.assert_close(seen[0], self.tokens.mean(dim=1, keepdim=True))

    def test_pooled_and_spatial_semantic_heads_match_for_constant_unpositioned_context(self):
        with torch.no_grad():
            pooled = self.head("B2").eval()
            spatial = self.head("J").eval()
            restored = spatial.load_state_dict(pooled.state_dict(), strict=False)
            self.assertEqual(restored.missing_keys, ["spatial_positions"])
            self.assertEqual(restored.unexpected_keys, [])
            spatial.spatial_positions.zero_()
            constant_tokens = self.tokens[:, :1].expand(-1, 16, -1)
            for task, logits in spatial.forward_all(constant_tokens).items():
                torch.testing.assert_close(logits, pooled(constant_tokens, task), rtol=1e-5, atol=1e-6)

    def test_custom_visual_token_width(self):
        for method in METHODS:
            with self.subTest(method=method), torch.no_grad():
                head = self.head(method, token_width=32)
                self.assertEqual(head(torch.randn(2, 16, 32), "age").shape, (2, 5))
                with self.assertRaises(ValueError):
                    head(self.tokens, "age")

    def test_invalid_tokens_tasks_methods_and_feature_tables(self):
        head = self.head("J")
        for tokens in (torch.randn(3, 256), torch.randn(3, 15, 256), torch.randn(3, 16, 128), torch.zeros(3, 16, 256, dtype=torch.long)):
            with self.subTest(shape=tokens.shape), self.assertRaises(ValueError):
                head(tokens, "age")
        with self.assertRaises(ValueError):
            head(self.tokens, "unknown")
        with self.assertRaises(ValueError):
            self.head("unknown")
        for width in (0, -1, 2.5, True):
            with self.subTest(width=width), self.assertRaises(ValueError):
                self.head("J", token_width=width)
        invalid_text = (
            {"geology": self.text["geology"]},
            {**self.text, "geology": torch.randn(48, 12)},
            {**self.text, "age": torch.randn(5, 13)},
            {**self.text, "age": torch.ones(5, 12, dtype=torch.long)},
            {**self.text, "age": torch.full((5, 12), torch.nan)},
        )
        for text in invalid_text:
            with self.assertRaises(ValueError):
                build_head("J", text, self.tasks)
        for task in ({"age": self.tasks["age"]}, {**self.tasks, "age": torch.randn(1, 12)}):
            with self.assertRaises(ValueError):
                build_head("J", self.text, task)

    def test_masked_probabilities_broadcast_exact_zeros_and_gradients(self):
        logits = torch.tensor([[1000., -1000., 2., 3.], [-1000., 1000., -2., -3.]], requires_grad=True)
        mask = torch.tensor([False, True, False, True])
        probabilities = masked_probabilities(logits, mask)
        self.assertTrue(torch.isfinite(probabilities).all())
        self.assertTrue(torch.equal(probabilities[:, ~mask], torch.zeros(2, 2)))
        torch.testing.assert_close(probabilities.sum(-1), torch.ones(2))
        torch.testing.assert_close(probabilities[:, mask], logits[:, mask].softmax(-1))
        (probabilities * torch.arange(4)).sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue(torch.equal(logits.grad[:, ~mask], torch.zeros(2, 2)))
        torch.testing.assert_close(masked_probabilities(logits), logits.softmax(-1))
        tensor = torch.randn(2, 3, 4)
        broadcast_mask = torch.tensor([[[True, False, False, True]], [[False, True, True, False]]])
        expanded = broadcast_mask.expand_as(tensor)
        actual = masked_probabilities(tensor, broadcast_mask)
        torch.testing.assert_close(actual.sum(-1), torch.ones(2, 3))
        self.assertTrue(torch.equal(actual[~expanded], torch.zeros_like(actual[~expanded])))

    def test_different_task_widths_can_share_a_padded_masked_batch(self):
        with torch.no_grad():
            for method in METHODS:
                with self.subTest(method=method):
                    outputs = self.head(method).forward_all(self.tokens)
                    padded = torch.zeros(6, 49)
                    padded[:3] = outputs["geology"]
                    padded[3:, :5] = outputs["age"]
                    mask = torch.arange(49)[None, :] < torch.tensor([49, 49, 49, 5, 5, 5])[:, None]
                    probabilities = masked_probabilities(padded, mask)
                    torch.testing.assert_close(probabilities[:3], outputs["geology"].softmax(-1))
                    torch.testing.assert_close(probabilities[3:, :5], outputs["age"].softmax(-1))
                    self.assertTrue(torch.equal(probabilities[3:, 5:], torch.zeros(3, 44)))

    def test_masked_probabilities_rejects_empty_rows_and_invalid_masks(self):
        logits = torch.randn(2, 5)
        for mask in (torch.zeros(5, dtype=torch.bool),
                     torch.tensor([[True] * 5, [False] * 5]),
                     torch.ones(5), torch.ones(3, dtype=torch.bool)):
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                masked_probabilities(logits, mask)
        for invalid in (torch.empty(2, 0), torch.tensor(1.), torch.ones(2, 5, dtype=torch.long)):
            with self.assertRaises(ValueError):
                masked_probabilities(invalid)


if __name__ == "__main__":
    unittest.main()
