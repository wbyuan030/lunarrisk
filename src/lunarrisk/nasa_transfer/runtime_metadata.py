"""Offline direct-distribution metadata audit, never runtime compatibility approval.

Reads installed metadata only. No package imports, resolver, artifact acquisition,
transitive dependency closure, ABI checks, model execution, or device inspection.
"""
from __future__ import annotations

import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform

from packaging.requirements import Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

NASA_REVISION = "d54c67aad513cb9daca444afa425cfb278e4fbf8"
PLAN_ID = "p33_private_nasa_runtime_candidate_v1"
PYTHON_REQUIREMENT = ">=3.11,<3.13"
SOURCE_AUDIT_SHA256 = "fe555785c530d9beda559d7ee8d2736379ae325d8223a17342458d1e96651e24"
REQUIREMENTS_SHA256 = "f9b4aea5cf5272114e4c8306fece9858aff0e909252a4cfc596b9a16b939d18f"
PLAN_PATH = "protocols/p33_private_runtime_plan.json"
SOURCE_AUDIT_PATH = "reports/evidence/p33_20261105/runtime_plan_01/source_audit.json"
# Exact declarations from the independently pinned source audit, not caller input.
NASA_REQUIREMENTS = (
    "terratorch>=1.2.6,<2", "torch>=2.12.0,<3", "torchvision>=0.27.0,<0.28",
    "torchmetrics>=1.9.0,<2", "lightning>=2.6.5,<3", "timm>=1.0.27,<2",
    "einops>=0.8.2,<0.9", "numpy>=2.4.4,<3", "omegaconf>=2.3.0,<3",
    "pillow>=12.2.0,<13", "rasterio>=1.5.0,<2", "torchgeo>=0.9.0,<0.10",
    "albumentations>=2.0,<3", "segmentation-models-pytorch>=0.4,<0.6",
    "tensorboard>=2.18,<3", "h5py>=3.16.0,<4", "hdf5plugin>=6.0.0,<7",
    "pandas>=3.0.3,<4", "pyarrow>=24.0.0,<25", "diffusers>=0.38.0,<0.39",
    "tokenizers>=0.22.2,<0.23", "vector-quantize-pytorch>=1.29.0,<2",
    "huggingface-hub>=1.15.0,<2", "peft>=0.19.1,<0.20",
    "scipy>=1.17.1,<2", "scikit-image>=0.26.0,<0.27",
    "matplotlib>=3.10.9,<4", "tqdm>=4.67.3,<5",
)
CPU_CANDIDATE = {"torch": "2.12.1+cpu", "torchvision": "0.27.1+cpu"}


def validate_evidence(plan, source_audit_bytes):
    """Reject edited declarations/revisions, even if both JSONs were edited alike."""
    if hashlib.sha256(source_audit_bytes).hexdigest() != SOURCE_AUDIT_SHA256:
        raise ValueError("source audit SHA256 differs from immutable pin")
    source = json.loads(source_audit_bytes)
    for field, expected in (("plan_id", PLAN_ID), ("nasa_code_revision", NASA_REVISION),
                            ("declared_requirements", list(NASA_REQUIREMENTS)),
                            ("source_audit", SOURCE_AUDIT_PATH)):
        if plan.get(field) != expected:
            raise ValueError("plan differs from pinned declaration: " + field)
    if (source.get("revision") != NASA_REVISION
            or source.get("declared_requirements") != list(NASA_REQUIREMENTS)
            or source.get("python_requirement") != PYTHON_REQUIREMENT):
        raise ValueError("source audit declaration mismatch")
    rows = [s for s in source["sources"] if s["path"] == "requirements.txt"]
    if len(rows) != 1 or rows[0]["sha256"] != REQUIREMENTS_SHA256:
        raise ValueError("upstream requirements hash mismatch")


def installed_records():
    """Enumerate metadata without importing the distributions' Python packages."""
    wanted = {canonicalize_name(Requirement(r).name) for r in NASA_REQUIREMENTS}
    records = []
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        if name and canonicalize_name(name) in wanted:
            records.append({"name": name, "version": dist.version,
                            "requires_python": dist.metadata.get("Requires-Python")})
    return records


def audit_metadata(records, python_version, *, cpu_candidate=False):
    """Evaluate injected metadata records against immutable direct requirements.

    Fixed NASA declarations contain no markers or extras; transitive Requires-Dist
    and their markers/extras are deliberately not evaluated.
    """
    python = Version(python_version)
    indexed = {}
    for record in records:
        indexed.setdefault(canonicalize_name(record["name"]), []).append(record)
    python_match = SpecifierSet(PYTHON_REQUIREMENT).contains(python, prereleases=False)
    candidate_python_match = python.release[:2] == (3, 12) and not python.is_prerelease and not python.is_devrelease
    checks = []
    for declared in NASA_REQUIREMENTS:
        requirement = Requirement(declared)
        name = canonicalize_name(requirement.name)
        matches = indexed.get(name, [])
        row = {"name": name, "requirement": declared, "installed": matches}
        if not matches:
            row["status"] = "MISSING"
        elif len(matches) != 1:
            row["status"] = "AMBIGUOUS_METADATA"
        else:
            record = matches[0]
            try:
                version = Version(record["version"])
                requires_python = record.get("requires_python")
                if not requirement.specifier.contains(version, prereleases=False):
                    row["status"] = "VERSION_CONFLICT"
                elif requires_python and not SpecifierSet(requires_python).contains(python, prereleases=False):
                    row["status"] = "REQUIRES_PYTHON_CONFLICT"
                elif cpu_candidate and name in CPU_CANDIDATE and version != Version(CPU_CANDIDATE[name]):
                    row["status"] = "CPU_CANDIDATE_VERSION_CONFLICT"
                else:
                    row["status"] = "METADATA_MATCH"
            except (InvalidVersion, InvalidSpecifier, TypeError):
                row["status"] = "INVALID_METADATA"
        checks.append(row)
    match = python_match and (not cpu_candidate or candidate_python_match)
    match = match and all(row["status"] == "METADATA_MATCH" for row in checks)
    return {
        "schema": "p33_runtime_direct_metadata_v1",
        "status": "DIRECT_METADATA_MATCH" if match else "DIRECT_METADATA_BLOCKED",
        "scope": "NASA28_DIRECT_DISTRIBUTIONS_ONLY",
        "nasa_code_revision": NASA_REVISION,
        "requirements_sha256": REQUIREMENTS_SHA256,
        "source_audit_sha256": SOURCE_AUDIT_SHA256,
        "python": {"version": python_version, "requirement": PYTHON_REQUIREMENT,
                   "metadata_match": python_match},
        "cpu_candidate": {"selected": cpu_candidate, "python_requirement": "3.12.x final release",
                          "python_match": candidate_python_match if cpu_candidate else None,
                          "exact_versions": CPU_CANDIDATE if cpu_candidate else {}},
        "distributions": checks,
        "compatibility": "NOT_ESTABLISHED",
        "upstream_execution": "NOT_RUN", "model_execution": "NOT_RUN",
        "dependency_closure": "NOT_CHECKED", "hardware": "NOT_INSPECTED",
        "limitations": ["Metadata does not prove import, ABI, wheel provenance, CPU execution, or model compatibility.",
                        "Transitive Requires-Dist/markers/extras and build requirements are not checked.",
                        "No installation, resolution, network, model/data access, or GPU inspection is performed."],
    }


def audit_repository(root, *, cpu_candidate=False):
    root = Path(root)
    plan = json.loads((root / PLAN_PATH).read_text(encoding="utf-8"))
    validate_evidence(plan, (root / SOURCE_AUDIT_PATH).read_bytes())
    return audit_metadata(installed_records(), platform.python_version(), cpu_candidate=cpu_candidate)
