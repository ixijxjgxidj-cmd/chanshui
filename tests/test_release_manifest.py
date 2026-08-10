"""生产发布清单与部署前校验器测试。"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_release_manifest.py"
SPEC = importlib.util.spec_from_file_location("verify_release_manifest", SCRIPT)
assert SPEC and SPEC.loader
verify = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = verify
SPEC.loader.exec_module(verify)

SANITIZER_SCRIPT = ROOT / "scripts" / "sanitize_artifact_metadata.py"
SANITIZER_SPEC = importlib.util.spec_from_file_location(
    "sanitize_artifact_metadata", SANITIZER_SCRIPT
)
assert SANITIZER_SPEC and SANITIZER_SPEC.loader
sanitizer = importlib.util.module_from_spec(SANITIZER_SPEC)
sys.modules[SANITIZER_SPEC.name] = sanitizer
SANITIZER_SPEC.loader.exec_module(sanitizer)


def _manifest():
    return json.loads(
        (ROOT / "deploy" / "production_release_manifest.json").read_text(
            encoding="utf-8"
        )
    )


def test_actual_release_manifest_and_all_present_assets_verify():
    result = verify.verify_manifest(
        _manifest(), repo_root=ROOT, require_external=True, check_git=True
    )
    assert result["pass"], result["errors"]
    assert result["tracked_assets"] == 14
    assert result["external_assets"] == 1


def test_manifest_contains_no_machine_specific_absolute_paths():
    assert verify.machine_specific_strings(_manifest()) == []


def test_historical_model_manifest_contains_no_machine_specific_paths():
    historical = json.loads(
        (ROOT / "weights" / "official_r1_to_r2" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert verify.machine_specific_strings(historical) == []


def test_artifact_metadata_sanitizer_keeps_only_portable_names():
    assert sanitizer.sanitize_string("/home/runner/data/train.hdf5") == "train.hdf5"
    assert sanitizer.sanitize_string(r"C:\Users\runner\data\round1.zip") == "round1.zip"
    assert sanitizer.sanitize_string("weights/model.pt") == "weights/model.pt"


def test_tampered_asset_hash_is_rejected():
    manifest = _manifest()
    manifest["assets"]["tracked"][0]["sha256"] = "0" * 64
    result = verify.verify_manifest(
        manifest, repo_root=ROOT, require_external=False, check_git=False
    )
    assert not result["pass"]
    assert any("SHA-256" in error for error in result["errors"])


def test_path_traversal_is_rejected_before_file_access():
    manifest = _manifest()
    manifest["assets"]["tracked"][0]["path"] = "../outside.bin"
    result = verify.verify_manifest(
        manifest, repo_root=ROOT, require_external=False, check_git=False
    )
    assert not result["pass"]
    assert any("越出仓库" in error for error in result["errors"])


def test_config_drift_between_manifest_and_deploy_script_is_rejected():
    manifest = _manifest()
    manifest["production_config"]["long_dedup_s"] = 30.0
    result = verify.verify_manifest(
        manifest, repo_root=ROOT, require_external=False, check_git=False
    )
    assert not result["pass"]
    assert any("long_dedup_s" in error for error in result["errors"])


def test_fallback_policy_drift_is_rejected():
    manifest = _manifest()
    manifest["deployment_policy"]["allow_model_fallback_default"] = True
    result = verify.verify_manifest(
        manifest, repo_root=ROOT, require_external=False, check_git=False
    )
    assert not result["pass"]
    assert any("allow_model_fallback_default" in error for error in result["errors"])


def test_missing_required_external_asset_is_rejected():
    manifest = _manifest()
    manifest["assets"]["external"][0]["path"] = (
        "weights/seismicxm/missing-for-test.pt"
    )
    result = verify.verify_manifest(
        manifest, repo_root=ROOT, require_external=True, check_git=False
    )
    assert not result["pass"]
    assert any("资产缺失" in error for error in result["errors"])


def test_deploy_verification_happens_before_service_restart():
    text = (ROOT / "deploy" / "deploy_api.sh").read_text(encoding="utf-8")
    verify_pos = text.index("verify_release_manifest.py")
    restart_pos = text.index("systemctl restart phasepick-api")
    assert verify_pos < restart_pos
    assert "ALLOW_MODEL_FALLBACK" in text
