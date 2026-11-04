"""Public, sample-independent taxonomy prototypes; no pretrained language semantics.

The literal-symbol token is deliberately case sensitive and identity-like. TF-IDF
therefore preserves all official classes, including symbols that collide when
lowercased. This tier tests lexical/taxonomy structure, not language understanding.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from lunarrisk.benchmarks.moonstone import AGE_NAMES, age_target

TIER = "TAXONOMY_ONLY_LOCAL"
TASKS = ("geology", "age")
OFFICIAL_SYMBOLS = tuple(
    "Cc Ccc Csc EIp Ec Ecc Em Esc INp INt Ib Ibm Ic Ic1 Ic2 Icc Icf Id Ig "
    "Iia Iiap Iic Iif Im1 Im2 Imd Iohi Ioho Iohs Iom Iork Iorm Ip Isc It Itd "
    "Nb Nbl Nbm Nbsc Nc Nnj Np Nt Ntp pNb pNbm pNc pNt".split()
)
TASK_DEFINITIONS = {
    "geology": "Choose the official Moonstone geology symbol for the lunar observation.",
    "age": "Choose the official Moonstone age category for the lunar observation.",
}
SOURCE_HASHES = {
    "geologic_units": "d7823219998184601e73a2445106a6483b88a4365384aee81fb3f343914562ba",
    "age_mapping": "34e184ee878dfb38944be49d2fc34c7a1438a36b7a06b778db8cc49c96209078",
}
SOURCE_PATHS = {
    "geologic_units": "reports/evidence/p30_20261103/geologic_units.json",
    "age_mapping": "src/lunarrisk/benchmarks/moonstone.py",
}
PERMUTATION_SEED = 20261104


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _expected_candidate(task, index):
    symbol = OFFICIAL_SYMBOLS[index] if task == "geology" else AGE_NAMES[index]
    age_id = age_target(symbol) if task == "geology" else index
    sources = ["geologic_units", "age_mapping"] if task == "geology" else ["age_mapping"]
    row = {
        "task": task, "class_id": index, "symbol": symbol, "public_name": symbol,
        "description": (f"Public geology symbol {symbol}. " if task == "geology" else "")
                       + f"Official Moonstone age category: {AGE_NAMES[age_id]}.",
        "source": sources, "source_hash": {key: SOURCE_HASHES[key] for key in sources},
        "status": TIER,
    }
    if task == "geology":
        row.update(official_unit_id=index + 1, age_id=age_id)
    return row


def _validate_catalog(catalog):
    # This version intentionally accepts only the frozen local taxonomy template.
    # New definitions/templates require a new version/tier and separate manifest.
    expected_keys = {"catalog_version", "tier", "shared_for_all_samples", "sources", "tasks",
                     "representation", "descriptor_control", "asset_inventory", "limitations"}
    if not isinstance(catalog, dict) or set(catalog) != expected_keys:
        raise ValueError("catalog schema mismatch; sample-specific fields are forbidden")
    if (catalog["catalog_version"] != "P32_CANDIDATE_CATALOG_V1"
            or catalog["tier"] != TIER or catalog["shared_for_all_samples"] is not True):
        raise ValueError("unsupported catalog version/tier or sample-dependent catalog")
    if set(catalog["sources"]) != set(SOURCE_HASHES):
        raise ValueError("unexpected public sources")
    for key, digest in SOURCE_HASHES.items():
        if (catalog["sources"][key]["sha256"] != digest
                or catalog["sources"][key]["path"] != SOURCE_PATHS[key]):
            raise ValueError("public source provenance mismatch")
    if set(catalog["tasks"]) != set(TASKS):
        raise ValueError("only the official geology and age tasks are supported")
    for task in TASKS:
        table = catalog["tasks"][task]
        count = 49 if task == "geology" else 5
        if (set(table) != {"definition", "candidates"}
                or table["definition"] != TASK_DEFINITIONS[task]
                or table["candidates"] != [_expected_candidate(task, i) for i in range(count)]):
            raise ValueError(f"{task}: unknown symbol, class order, mapping, or taxonomy text")
    if catalog["representation"] != {
        "algorithm": "catalog_word_literal_tfidf_v1", "fit_scope": "54 candidates and 2 task definitions only",
        "lowercase_words": True, "case_sensitive_literal_symbols": True,
        "term_frequency": "raw count", "idf": "log((1 + N) / (1 + document_frequency)) + 1",
        "normalization": "L2 per row", "dtype": "float32", "pretrained_language_model": False,
    }:
        raise ValueError("representation differs from the fixed lexical protocol")
    if catalog["descriptor_control"] != {
        "seed": PERMUTATION_SEED, "algorithm": "numpy PCG64 sequential per-task permutation; geology then age",
        "mapping": "destination class row i receives entire source descriptor row permutation[task][i]",
        "same_mapping_train_and_inference": True, "task_vectors_permuted": False,
        "prompt_search": False,
    }:
        raise ValueError("descriptor control differs from the predeclared protocol")
    # Enforce JSON metadata now rather than permitting numpy objects or nonfinite values.
    _json_hash(catalog)


def load_catalog(path):
    """Load the frozen public table and verify local audited source bytes (no network)."""
    catalog = json.loads(Path(path).read_text(encoding="utf-8"))
    _validate_catalog(catalog)
    root = Path(__file__).resolve().parents[3]
    for key, relative in SOURCE_PATHS.items():
        digest = hashlib.sha256()
        with (root / relative).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != SOURCE_HASHES[key]:
            raise ValueError(f"audited source SHA256 mismatch: {key}")
    mapping = json.loads((root / SOURCE_PATHS["geologic_units"]).read_text(encoding="utf-8"))
    if mapping != {symbol: i + 1 for i, symbol in enumerate(OFFICIAL_SYMBOLS)}:
        raise ValueError("official geology source mapping mismatch")
    return catalog


def _tokens(text, task=None, symbol=None):
    tokens = ["word:" + token.lower() for token in re.findall(r"[A-Za-z0-9]+", text)]
    if symbol is not None:
        tokens.append(f"literal:{task}:{symbol}")
    return tokens


def encode_catalog(catalog):
    """Return fixed FP32 [K,D] candidates and [D] tasks, plus JSON provenance.

    Vocabulary/IDF use exactly 54 public candidate documents plus 2 task
    definitions. No sample, target, split, image or validation input is accepted.
    """
    _validate_catalog(catalog)
    documents, descriptors = [], {}
    for task in TASKS:
        descriptors[task] = deepcopy(catalog["tasks"][task]["candidates"])
        for row in descriptors[task]:
            documents.append(_tokens(row["description"], task, row["symbol"]))
    documents.extend(_tokens(catalog["tasks"][task]["definition"]) for task in TASKS)
    vocab = sorted({token for document in documents for token in document})
    lookup = {token: index for index, token in enumerate(vocab)}
    frequency = Counter(token for document in documents for token in set(document))
    idf = np.asarray([np.log((1 + len(documents)) / (1 + frequency[token])) + 1
                      for token in vocab], dtype=np.float64)
    matrix = np.zeros((len(documents), len(vocab)), dtype=np.float64)
    for index, document in enumerate(documents):
        for token, count in Counter(document).items():
            column = lookup[token]
            matrix[index, column] = count * idf[column]
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix.astype(np.float32)
    return {
        "candidate_features": {"geology": matrix[:49].copy(), "age": matrix[49:54].copy()},
        "task_features": {task: matrix[54 + i].copy() for i, task in enumerate(TASKS)},
        "geology_to_age": [age_target(symbol) for symbol in OFFICIAL_SYMBOLS],
        "metadata": {
            "tier": TIER, "algorithm": catalog["representation"]["algorithm"],
            "representation": deepcopy(catalog["representation"]), "vocab": vocab,
            "idf": idf.tolist(), "document_count": len(documents),
            "catalog_sha256": _json_hash(catalog), "catalog_hash_format": "canonical sorted compact UTF-8 JSON",
            "source_hashes": dict(SOURCE_HASHES), "sources": deepcopy(catalog["sources"]),
            "descriptors": descriptors,
            "task_definitions": {task: catalog["tasks"][task]["definition"] for task in TASKS},
            "descriptor_alignment": "aligned", "permutation": {task: list(range(len(descriptors[task]))) for task in TASKS},
            "descriptor_control": deepcopy(catalog["descriptor_control"]),
            "asset_inventory": deepcopy(catalog["asset_inventory"]),
            "limitations": deepcopy(catalog["limitations"]),
        },
    }


def permute_encoded(encoded, seed=PERMUTATION_SEED):
    """Copy aligned encodings and permute full descriptor rows within each task.

    Output class slots and geology_to_age remain the official target mapping.
    metadata.descriptors and features move together, including the literal token
    and Age knowledge. Task vectors and shared vocabulary/IDF stay fixed. Use
    this same predeclared mapping during both training and inference.
    """
    if encoded["metadata"]["descriptor_alignment"] != "aligned":
        raise ValueError("permute_encoded expects aligned input, not an already permuted table")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    result = deepcopy(encoded)
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    for task in TASKS:
        count = 49 if task == "geology" else 5
        if encoded["candidate_features"][task].shape[0] != count:
            raise ValueError("candidate row count mismatch")
        permutation = rng.permutation(count).tolist()
        result["candidate_features"][task] = encoded["candidate_features"][task][permutation].copy()
        result["metadata"]["descriptors"][task] = [
            deepcopy(encoded["metadata"]["descriptors"][task][i]) for i in permutation]
        result["metadata"]["permutation"][task] = permutation
    result["metadata"]["descriptor_alignment"] = "fixed_permutation"
    result["metadata"]["permutation_seed"] = int(seed)
    return result
