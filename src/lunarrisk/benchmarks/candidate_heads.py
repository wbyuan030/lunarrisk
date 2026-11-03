"""Small same-feature controls, not a pretrained or semantic class-description model."""
import torch
from torch import nn


class OrdinaryHead(nn.Module):
    def __init__(self, feature_dim=256, classes=49):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(feature_dim, 40), nn.ReLU(), nn.Linear(40, classes))

    def forward(self, features):
        return self.layers(features)


class CandidateIDHead(nn.Module):
    """Score each public class ID using one shared MLP. No per-sample target enters the scorer.

    With 256-D features and 49 classes this has 12,293 parameters versus OrdinaryHead's 12,289.
    Shared candidate lists may be permuted/subsetted; this does not imply unseen-class generalization.
    """
    def __init__(self, feature_dim=256, classes=49):
        super().__init__()
        self.classes = classes
        self.embedding = nn.Embedding(classes, 16)
        self.scorer = nn.Sequential(nn.Linear(feature_dim+16, 42), nn.ReLU(), nn.Linear(42, 1))

    def forward(self, features, candidate_ids):
        if (candidate_ids.ndim != 1 or candidate_ids.dtype != torch.long or
                candidate_ids.numel() == 0 or candidate_ids.unique().numel() != candidate_ids.numel()):
            raise ValueError("one shared nonempty unique int64 candidate list required")
        if ((candidate_ids < 0) | (candidate_ids >= self.classes)).any():
            raise ValueError("candidate ID outside the public class map")
        candidate = self.embedding(candidate_ids)
        b, k = features.shape[0], len(candidate_ids)
        joined = torch.cat((features[:, None, :].expand(b, k, -1),
                            candidate[None, :, :].expand(b, k, -1)), dim=-1)
        return self.scorer(joined).squeeze(-1)


def masked_softmax(logits, allowed):
    """Ordinary-classifier subset control; reject empty candidate sets instead of returning NaN."""
    if allowed.shape != logits.shape or allowed.dtype != torch.bool or not allowed.any(dim=-1).all():
        raise ValueError("each row requires a same-shape nonempty boolean candidate mask")
    return torch.softmax(logits.masked_fill(~allowed, float("-inf")), dim=-1)
