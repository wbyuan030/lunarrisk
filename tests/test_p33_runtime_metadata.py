"""Synthetic, offline direct metadata tests; no official runtime execution."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lunarrisk.nasa_transfer import runtime_metadata as audit


def fixture():
    records = []
    for declared in audit.NASA_REQUIREMENTS:
        requirement = audit.Requirement(declared)
        minimum = next(spec.version for spec in requirement.specifier if spec.operator == ">=")
        records.append({"name": requirement.name, "version": minimum, "requires_python": ">=3.11"})
    return records


def status(result, name):
    return next(r["status"] for r in result["distributions"] if r["name"] == name)


class TestRuntimeMetadata(unittest.TestCase):
    def test_all_direct_match_is_not_runtime_approval(self):
        result = audit.audit_metadata(fixture(), "3.12.8")
        self.assertEqual(result["status"], "DIRECT_METADATA_MATCH")
        for key in ("upstream_execution", "model_execution"):
            self.assertEqual(result[key], "NOT_RUN")
        self.assertEqual(result["compatibility"], "NOT_ESTABLISHED")
        self.assertEqual(result["dependency_closure"], "NOT_CHECKED")
        self.assertEqual(len(result["distributions"]), 28)

    def test_missing(self):
        result = audit.audit_metadata([], "3.12.8")
        self.assertEqual(result["status"], "DIRECT_METADATA_BLOCKED")
        self.assertTrue(all(r["status"] == "MISSING" for r in result["distributions"]))

    def test_version_conflict(self):
        rows = fixture()
        rows[1]["version"] = "2.5.1+rocm6.2"
        self.assertEqual(status(audit.audit_metadata(rows, "3.12.8"), "torch"), "VERSION_CONFLICT")

    def test_local_cpu_version_and_candidate(self):
        rows = fixture()
        rows[1]["version"] = "2.12.1+cpu"
        rows[2]["version"] = "0.27.1+cpu"
        for candidate in (False, True):
            result = audit.audit_metadata(rows, "3.12.8", cpu_candidate=candidate)
            self.assertEqual(result["status"], "DIRECT_METADATA_MATCH")
        for wrong in ("2.12.1", "2.12.1+rocm7.2", "2.12.2+cpu"):
            rows[1]["version"] = wrong
            result = audit.audit_metadata(rows, "3.12.8", cpu_candidate=True)
            self.assertEqual(status(result, "torch"), "CPU_CANDIDATE_VERSION_CONFLICT")

    def test_python_bounds_and_candidate(self):
        for version in ("3.10.12", "3.13.0", "3.12.0rc1"):
            self.assertEqual(audit.audit_metadata(fixture(), version)["status"], "DIRECT_METADATA_BLOCKED")
        result = audit.audit_metadata(fixture(), "3.11.9", cpu_candidate=True)
        self.assertTrue(result["python"]["metadata_match"])
        self.assertFalse(result["cpu_candidate"]["python_match"])
        self.assertEqual(result["status"], "DIRECT_METADATA_BLOCKED")

    def test_prerelease_distribution_is_not_approved(self):
        rows = fixture()
        rows[1]["version"] = "2.13.0rc1"
        self.assertEqual(status(audit.audit_metadata(rows, "3.12.8"), "torch"), "VERSION_CONFLICT")

    def test_malformed_python_version_fails_closed(self):
        for version in ("not-python", "", "3.12..8"):
            with self.subTest(version=version), self.assertRaises(ValueError):
                audit.audit_metadata(fixture(), version)

    def test_torchvision_cpu_candidate_mismatch(self):
        rows = fixture()
        rows[1]["version"] = "2.12.1+cpu"
        for wrong in ("0.27.1", "0.27.1+rocm7.2", "0.27.2+cpu"):
            with self.subTest(version=wrong):
                rows[2]["version"] = wrong
                result = audit.audit_metadata(rows, "3.12.8", cpu_candidate=True)
                self.assertEqual(status(result, "torchvision"), "CPU_CANDIDATE_VERSION_CONFLICT")
                self.assertEqual(result["status"], "DIRECT_METADATA_BLOCKED")

    def test_missing_version_or_name_fails_closed(self):
        for field in ("version", "name"):
            rows = fixture()
            del rows[0][field]
            with self.subTest(field=field), self.assertRaises(KeyError):
                audit.audit_metadata(rows, "3.12.8")

    def test_installed_requires_python(self):
        rows = fixture()
        rows[0]["requires_python"] = ">=3.13"
        self.assertEqual(status(audit.audit_metadata(rows, "3.12.8"), "terratorch"), "REQUIRES_PYTHON_CONFLICT")

    def test_invalid_metadata(self):
        for field, value in (("version", "garbage"), ("requires_python", "not a specifier")):
            rows = fixture()
            rows[0][field] = value
            self.assertEqual(status(audit.audit_metadata(rows, "3.12.8"), "terratorch"), "INVALID_METADATA")

    def test_duplicates_are_not_silently_selected(self):
        rows = fixture()
        rows.append(dict(rows[0], name="TerraTorch"))
        self.assertEqual(status(audit.audit_metadata(rows, "3.12.8"), "terratorch"), "AMBIGUOUS_METADATA")

    def test_pinned_evidence_and_edited_declarations(self):
        plan = json.loads((ROOT / audit.PLAN_PATH).read_text())
        source = (ROOT / audit.SOURCE_AUDIT_PATH).read_bytes()
        audit.validate_evidence(plan, source)
        for field, value in (("nasa_code_revision", "changed"), ("plan_id", "changed"),
                             ("declared_requirements", ["torch>=2"])):
            changed = dict(plan, **{field: value})
            with self.assertRaises(ValueError):
                audit.validate_evidence(changed, source)
        changed_source = json.loads(source)
        changed_source["declared_requirements"] = ["torch>=2"]
        with self.assertRaises(ValueError):
            audit.validate_evidence(dict(plan, declared_requirements=["torch>=2"]), json.dumps(changed_source).encode())

    def test_source_audit_path_mutation_rejected(self):
        plan = json.loads((ROOT / audit.PLAN_PATH).read_text())
        plan["source_audit"] = "reports/unpinned_audit.json"
        with self.assertRaisesRegex(ValueError, "source_audit"):
            audit.validate_evidence(plan, (ROOT / audit.SOURCE_AUDIT_PATH).read_bytes())

    def test_marker_changes_rejected_not_silently_skipped(self):
        plan = json.loads((ROOT / audit.PLAN_PATH).read_text())
        plan["declared_requirements"][0] += '; python_version < "3.0"'
        with self.assertRaises(ValueError):
            audit.validate_evidence(plan, (ROOT / audit.SOURCE_AUDIT_PATH).read_bytes())

    def test_collection_reads_only_distribution_metadata(self):
        class Dist:
            metadata = {"Name": "Torch", "Requires-Python": ">=3.11"}
            version = "2.12.1+cpu"
        with patch.object(audit.metadata, "distributions", return_value=[Dist()]):
            self.assertEqual(audit.installed_records(), [{"name": "Torch", "version": "2.12.1+cpu", "requires_python": ">=3.11"}])

    def run_cli(self, *args):
        # Enforce no forbidden imports, socket use, or subprocess/network helpers
        # in the audited process, rather than merely checking a claim in JSON.
        code = '''import sys, runpy
blocked = {"torch", "torchvision", "ni_lfm", "terratorch", "numpy", "cupy", "jax", "requests", "urllib.request", "pip"}
def guard(event, args):
    if event == "import" and (args[0].split(".")[0] in blocked or args[0] in blocked):
        raise RuntimeError("forbidden import: " + args[0])
    if event.startswith("socket.") or event.startswith("subprocess.") or event in ("os.system", "os.posix_spawn"):
        raise RuntimeError("forbidden execution: " + event)
sys.addaudithook(guard)
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
'''
        return subprocess.run([sys.executable, "-c", code, str(ROOT / "scripts/p33_runtime_metadata.py"), *args],
                              capture_output=True, text=True, timeout=30)

    def test_cli_no_package_imports_or_network(self):
        result = self.run_cli("--cpu-candidate")
        self.assertIn(result.returncode, (0, 1), result.stderr + result.stdout)
        report = json.loads(result.stdout)
        self.assertEqual(report["model_execution"], "NOT_RUN")
        self.assertEqual(report["hardware"], "NOT_INSPECTED")

    def test_cli_exclusive_output(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "audit.json"
            first = self.run_cli("--output", str(path))
            self.assertIn(first.returncode, (0, 1), first.stderr + first.stdout)
            original = path.read_bytes()
            self.assertEqual(json.loads(original), json.loads(first.stdout))
            second = self.run_cli("--output", str(path))
            self.assertEqual(second.returncode, 2)
            self.assertEqual(json.loads(second.stdout)["status"], "AUDIT_ERROR")
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
