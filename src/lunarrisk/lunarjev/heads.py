"""P32 heads over shared, externally encoded 4x4 visual contexts.

B0 is an ordinary pooled classifier, B1 a pooled class-ID control, B2 a
pooled descriptor control, and J a spatial descriptor-query head. B1/B2/J
share the same two-block cross-attention architecture; the pooled controls
supply one context token. Candidate queries never attend to each other, so
subset logits equal the corresponding full-set logits up to floating-point
roundoff. This identity is an interface property, not semantic evidence.
"""

from collections.abc import Mapping
import math
from numbers import Integral

import torch
from torch import nn


TASK_CLASSES = {"geology": 49, "age": 5}
HIDDEN_WIDTH = 128
ATTENTION_HEADS = 4
FEEDFORWARD_WIDTH = 256
BLOCKS = 2
TOKEN_COUNT = 16
_INTEGER_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)


def masked_probabilities(logits, mask=None):
    """Softmax over the last axis, with exactly zero mass on masked entries.

    A boolean mask may broadcast to the logits' shape (including a shared
    candidate mask or per-row padding). Every broadcast row must allow at
    least one candidate. The caller supplies logits for the same candidate
    order as the mask; this function does not select or reorder candidates.
    """
    if not isinstance(logits, torch.Tensor) or not logits.is_floating_point():
        raise ValueError("logits must be a floating-point tensor")
    if logits.ndim < 1 or logits.shape[-1] == 0:
        raise ValueError("logits need a nonempty candidate axis")
    if mask is None:
        return logits.softmax(dim=-1)
    mask = torch.as_tensor(mask, device=logits.device)
    if mask.dtype != torch.bool:
        raise ValueError("mask must be boolean")
    try:
        mask = torch.broadcast_to(mask, logits.shape)
    except RuntimeError as exc:
        raise ValueError("mask must broadcast to the logits shape") from exc
    if not mask.any(dim=-1).all().item():
        raise ValueError("each row must allow at least one candidate")
    probabilities = logits.masked_fill(~mask, -torch.inf).softmax(dim=-1)
    return probabilities.masked_fill(~mask, 0.0)


def _candidate_ids(candidate_ids, classes, device):
    if candidate_ids is None:
        return torch.arange(classes, device=device)
    if isinstance(candidate_ids, torch.Tensor):
        ids = candidate_ids
        if ids.ndim != 1 or ids.numel() == 0:
            raise ValueError("candidate_ids must be a nonempty one-dimensional list")
        if ids.dtype not in _INTEGER_DTYPES:
            raise ValueError("candidate_ids must contain integers, not booleans or floats")
        ids = ids.to(device=device, dtype=torch.long)
    else:
        try:
            values = list(candidate_ids)
        except TypeError as exc:
            raise ValueError("candidate_ids must be a one-dimensional integer list") from exc
        if not values or any(isinstance(v, bool) or not isinstance(v, Integral) for v in values):
            raise ValueError("candidate_ids must be nonempty and contain only integers")
        if any(v < 0 or v >= classes for v in values):
            raise ValueError("candidate ID outside the task's class map")
        ids = torch.tensor(values, device=device, dtype=torch.long)
    if ((ids < 0) | (ids >= classes)).any().item():
        raise ValueError("candidate ID outside the task's class map")
    if ids.unique().numel() != ids.numel():
        raise ValueError("candidate_ids must not contain duplicates")
    return ids


def _validated_features(text_features, task_features):
    if not isinstance(text_features, Mapping) or not isinstance(task_features, Mapping):
        raise ValueError("text_features and task_features must be task mappings")
    if set(text_features) != set(TASK_CLASSES) or set(task_features) != set(TASK_CLASSES):
        raise ValueError("feature mappings must contain exactly geology and age")
    frozen = {}
    feature_width = None
    for task, classes in TASK_CLASSES.items():
        text, definition = text_features[task], task_features[task]
        if not isinstance(text, torch.Tensor) or not isinstance(definition, torch.Tensor):
            raise ValueError("text and task features must be floating-point tensors")
        if not text.is_floating_point() or not definition.is_floating_point():
            raise ValueError("text and task features must be floating-point tensors")
        if text.ndim != 2 or text.shape[0] != classes or text.shape[1] == 0:
            raise ValueError(f"{task} text_features must have shape [{classes}, D], D > 0")
        width = text.shape[1]
        if definition.shape != (width,):
            raise ValueError(f"{task} task_features must have shape [{width}]")
        if feature_width is not None and width != feature_width:
            raise ValueError("both tasks must use the same frozen feature width D")
        if not text.isfinite().all().item() or not definition.isfinite().all().item():
            raise ValueError("text and task features must be finite")
        feature_width = width
        # Own the storage: neither gradients nor later edits to caller tensors
        # may change this model's public descriptor table.
        frozen[task] = (text.detach().clone().float(), definition.detach().clone().float())
    return frozen, feature_width


class _Head(nn.Module):
    """Common validation, frozen public features, and two-task interface."""

    def __init__(self, text_features, task_features, token_width):
        super().__init__()
        if isinstance(token_width, bool) or not isinstance(token_width, Integral) or token_width <= 0:
            raise ValueError("token_width must be a positive integer")
        self.token_width = int(token_width)
        frozen, self.feature_width = _validated_features(text_features, task_features)
        for task, (text, definition) in frozen.items():
            self.register_buffer(f"text_{task}", text)
            self.register_buffer(f"task_{task}", definition)

    def _validate_tokens(self, tokens):
        if not isinstance(tokens, torch.Tensor) or not tokens.is_floating_point():
            raise ValueError("tokens must be a floating-point tensor")
        if tokens.ndim != 3 or tokens.shape[1:] != (TOKEN_COUNT, self.token_width):
            raise ValueError(f"tokens must have shape [B, {TOKEN_COUNT}, {self.token_width}]")

    def forward(self, tokens, task, candidate_ids=None):
        """Return [B, selected candidates] logits in the requested ID order."""
        if task not in TASK_CLASSES:
            raise ValueError(f"unknown task {task!r}; expected geology or age")
        self._validate_tokens(tokens)
        ids = _candidate_ids(candidate_ids, TASK_CLASSES[task], tokens.device)
        return self._score_context(self._encode_tokens(tokens), task, ids)

    def forward_all(self, tokens):
        """Return geology [B,49] and age [B,5], reusing prepared context."""
        self._validate_tokens(tokens)
        context = self._encode_tokens(tokens)
        return {
            task: self._score_context(context, task, torch.arange(classes, device=tokens.device))
            for task, classes in TASK_CLASSES.items()
        }


class _FeedForwardBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(HIDDEN_WIDTH)
        self.layers = nn.Sequential(
            nn.Linear(HIDDEN_WIDTH, FEEDFORWARD_WIDTH),
            nn.GELU(),
            nn.Linear(FEEDFORWARD_WIDTH, HIDDEN_WIDTH),
        )

    def forward(self, features):
        return features + self.layers(self.norm(features))


class OrdinaryHead(_Head):
    """B0: two ordinary task MLPs on the mean of the same visual tokens."""

    def __init__(self, text_features, task_features, token_width=256):
        super().__init__(text_features, task_features, token_width)
        self.task_heads = nn.ModuleDict({
            task: nn.Sequential(
                nn.Linear(self.token_width, HIDDEN_WIDTH),
                nn.GELU(),
                *[_FeedForwardBlock() for _ in range(BLOCKS)],
                nn.Linear(HIDDEN_WIDTH, classes),
            )
            for task, classes in TASK_CLASSES.items()
        })

    def _encode_tokens(self, tokens):
        return tokens.mean(dim=1)

    def _score_context(self, context, task, ids):
        return self.task_heads[task](context).index_select(-1, ids)


class _CrossAttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.query_norm = nn.LayerNorm(HIDDEN_WIDTH)
        self.context_norm = nn.LayerNorm(HIDDEN_WIDTH)
        self.attention = nn.MultiheadAttention(
            HIDDEN_WIDTH, ATTENTION_HEADS, dropout=0.0, batch_first=True,
        )
        self.feedforward = _FeedForwardBlock()

    def forward(self, queries, context):
        normalized = self.context_norm(context)
        attended, _ = self.attention(
            self.query_norm(queries), normalized, normalized, need_weights=False,
        )
        return self.feedforward(queries + attended)


def _fixed_2d_positions():
    """Row-major 4x4 sine/cosine positions; no learned or candidate positions."""
    quarter = HIDDEN_WIDTH // 4
    frequency = torch.exp(-math.log(10000.0) * torch.arange(quarter).float() / quarter)
    rows, columns = torch.meshgrid(torch.arange(4), torch.arange(4), indexing="ij")
    row_angles = rows.reshape(-1, 1) * frequency
    column_angles = columns.reshape(-1, 1) * frequency
    return torch.cat((row_angles.sin(), row_angles.cos(),
                      column_angles.sin(), column_angles.cos()), dim=-1)


class _QueryHead(_Head):
    def __init__(self, text_features, task_features, token_width, *, spatial, semantic):
        super().__init__(text_features, task_features, token_width)
        self.spatial = spatial
        self.semantic = semantic
        self.visual_projection = nn.Linear(self.token_width, HIDDEN_WIDTH)
        if semantic:
            # Shared across classes AND tasks. The only class-indexed state is
            # the frozen descriptor table; there is no trainable ID shortcut.
            self.text_projection = nn.Linear(self.feature_width, HIDDEN_WIDTH)
            self.task_projection = nn.Linear(self.feature_width, HIDDEN_WIDTH)
        else:
            self.candidate_embeddings = nn.ModuleDict({
                task: nn.Embedding(classes, HIDDEN_WIDTH)
                for task, classes in TASK_CLASSES.items()
            })
            self.task_embeddings = nn.Embedding(len(TASK_CLASSES), HIDDEN_WIDTH)
        self.blocks = nn.ModuleList([_CrossAttentionBlock() for _ in range(BLOCKS)])
        self.scorer = nn.Sequential(nn.LayerNorm(HIDDEN_WIDTH), nn.Linear(HIDDEN_WIDTH, 1))
        if spatial:
            self.register_buffer("spatial_positions", _fixed_2d_positions())

    def _encode_tokens(self, tokens):
        if self.spatial:
            return self.visual_projection(tokens) + self.spatial_positions
        return self.visual_projection(tokens.mean(dim=1, keepdim=True))

    def _score_context(self, context, task, ids):
        if self.semantic:
            descriptors = getattr(self, f"text_{task}").index_select(0, ids)
            query = self.text_projection(descriptors) + self.task_projection(getattr(self, f"task_{task}"))
        else:
            task_index = tuple(TASK_CLASSES).index(task)
            query = self.candidate_embeddings[task](ids) + self.task_embeddings.weight[task_index]
        queries = query.unsqueeze(0).expand(context.shape[0], -1, -1)
        for block in self.blocks:
            queries = block(queries, context)
        return self.scorer(queries).squeeze(-1)


class CandidateIDHead(_QueryHead):
    """B1: pooled context, learned known-class/task IDs, shared scorer."""

    def __init__(self, text_features, task_features, token_width=256):
        super().__init__(text_features, task_features, token_width, spatial=False, semantic=False)


class PooledSemanticHead(_QueryHead):
    """B2: same frozen task/descriptors and scorer as J, one pooled token."""

    def __init__(self, text_features, task_features, token_width=256):
        super().__init__(text_features, task_features, token_width, spatial=False, semantic=True)


class SpatialQueryHead(_QueryHead):
    """J: independent descriptor/task queries over 16 positioned visual tokens."""

    def __init__(self, text_features, task_features, token_width=256):
        super().__init__(text_features, task_features, token_width, spatial=True, semantic=True)


def build_head(method, text_features, task_features, token_width=256):
    """Build B0/B1/B2/J; features are frozen float tables for geology49/age5.

    text_features maps task -> [K,D], task_features maps task -> [D]. Both
    tasks use the same D from one fixed representation. Construction copies
    and detaches these tensors into persistent buffers (also in B0/B1 for
    provenance, though those controls do not use descriptor values).

    Call forward(tokens[B,16,token_width], task, candidate_ids=None) for logits,
    or forward_all(tokens) for both tasks. candidate_ids is a shared 1D unique
    integer sequence/tensor, with order preserved. No visual encoder is built
    or called here. Move the resulting module and input tensors to the same
    device/dtype using standard torch APIs.
    """
    implementations = {
        "B0": OrdinaryHead,
        "B1": CandidateIDHead,
        "B2": PooledSemanticHead,
        "J": SpatialQueryHead,
    }
    if method not in implementations:
        raise ValueError(f"unknown head method {method!r}; expected B0, B1, B2, or J")
    return implementations[method](text_features, task_features, token_width)
