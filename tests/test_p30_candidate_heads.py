"""Synthetic candidate/control contracts, not lunar classification evidence."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
try:
    import torch
    from lunarrisk.benchmarks.candidate_heads import OrdinaryHead, CandidateIDHead, masked_softmax
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch unavailable")
class TestCandidateHeads(unittest.TestCase):
    def test_permutation_and_subset_follow_public_candidate_ids(self):
        torch.manual_seed(42)
        head = CandidateIDHead().eval()
        x = torch.randn(3, 256)
        all_ids = torch.arange(49)
        subset = torch.tensor([48, 3, 0, 29])
        with torch.no_grad():
            reference = head(x, all_ids)
            actual = head(x, subset)
        torch.testing.assert_close(actual, reference[:, subset], rtol=1e-6, atol=1e-6)
        self.assertEqual(tuple(actual.shape), (3, 4))

    def test_masked_softmax_matches_subset_normalization_and_rejects_empty_rows(self):
        logits = torch.tensor([[1., 2., 4.], [5., 1., 0.]])
        allowed = torch.tensor([[True, False, True], [True, False, True]])
        actual = masked_softmax(logits, allowed)
        torch.testing.assert_close(actual[:, [0, 2]], torch.softmax(logits[:, [0, 2]], dim=-1))
        self.assertTrue((actual[:, 1] == 0).all())
        for mask in (torch.zeros_like(allowed), allowed.float(), allowed[:, :2]):
            with self.assertRaises(ValueError):
                masked_softmax(logits, mask)

    def test_invalid_or_duplicate_candidate_ids_fail(self):
        head, x = CandidateIDHead(), torch.zeros(1, 256)
        for ids in (torch.tensor([], dtype=torch.long), torch.tensor([0, 0]), torch.tensor([-1]),
                    torch.tensor([49]), torch.tensor([0.]), torch.tensor([[0, 1]])):
            with self.assertRaises(ValueError):
                head(x, ids)

    def test_parameter_budget_and_full_output_dimension_are_explicit(self):
        ordinary, candidate = OrdinaryHead(), CandidateIDHead()
        self.assertEqual(sum(p.numel() for p in ordinary.parameters()), 12289)
        self.assertEqual(sum(p.numel() for p in candidate.parameters()), 12293)
        self.assertEqual(tuple(ordinary(torch.zeros(2, 256)).shape), (2, 49))
        self.assertEqual(tuple(candidate(torch.zeros(2, 256), torch.arange(49)).shape), (2, 49))


if __name__ == "__main__":
    unittest.main()
