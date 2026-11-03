"""P31 candidate-ID structural identity; no semantic or accuracy claim."""
import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
try:
    import torch
    from lunarrisk.benchmarks.candidate_heads import CandidateIDHead,masked_softmax
except ImportError:
    torch=None


@unittest.skipIf(torch is None,"requires existing torch")
class TestCandidateControl(unittest.TestCase):
    def test_subset_softmax_equals_own_full_softmax_mask(self):
        torch.manual_seed(42)
        head=CandidateIDHead().eval()
        features=torch.randn(7,256)
        full=head(features,torch.arange(49))
        for subset in ([0],[48,0,7],[19,3,4,30,1],list(reversed(range(49)))):
            with self.subTest(subset=subset):
                ids=torch.tensor(subset)
                mask=torch.zeros_like(full,dtype=torch.bool)
                mask[:,ids]=True
                actual=head(features,ids).softmax(-1)
                expected=masked_softmax(full,mask)[:,ids]
                torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-7)


if __name__=="__main__": unittest.main()
