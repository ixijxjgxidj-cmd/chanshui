#!/usr/bin/env python3
"""校验生产发布清单、资产指纹和部署默认值。

默认允许外置 SeismicXM 编码器尚未复制到普通开发 clone；部署脚本使用
``--require-external``，在重启服务前强制其存在且哈希正确。
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "deploy" / "production_release_manifest.json"


SERVE_ARGUMENT_TO_CONFIG = {
    "--pretrained": "pretrained",
    "--p-threshold": "p_threshold",
    "--s-threshold": "s_threshold",
    "--p-merge-window": "p_merge_window_s",
    "--s-merge-window": "s_merge_window_s",
    "--cap-short-s": "cap_short_s",
    "--cap-max-p": "cap_max_p",
    "--cap-max-s": "cap_max_s",
    "--long-snr-db": "long_snr_db",
    "--long-snr-min-s": "long_snr_min_s",
    "--force-pair-short-s": "force_pair_short_s",
    "--force-pair-floor": "force_pair_floor",
    "--force-pair-mode": "force_pair_mode",
    "--long-dedup-s": "long_dedup_s",
    "--ensemble-long-members": "ensemble_long_members",
    "--mag-model": "mag_model",
    "--cls-model": "cls_model",
}

DEPLOY_DEFAULT_PATTERNS = {
    "pretrained": (r'^PRETRAINED="\$\{PRETRAINED:-(.*)\}"$', str),
    "weights_cli": (r'^PRODUCTION_WEIGHTS="(.*)"$', str),
    "cap_short_s": (r'^CAP_SHORT_S="\$\{CAP_SHORT_S:-(.*)\}"$', float),
    "cap_max_p": (r'^CAP_MAX_P="\$\{CAP_MAX_P:-(.*)\}"$', int),
    "cap_max_s": (r'^CAP_MAX_S="\$\{CAP_MAX_S:-(.*)\}"$', int),
    "long_snr_db": (r'^LONG_SNR_DB="\$\{LONG_SNR_DB:-(.*)\}"$', float),
    "long_snr_min_s": (
        r'^LONG_SNR_MIN_S="\$\{LONG_SNR_MIN_S:-(.*)\}"$',
        float,
    ),
    "force_pair_short_s": (
        r'^FORCE_PAIR_SHORT_S="\$\{FORCE_PAIR_SHORT_S:-(.*)\}"$',
        float,
    ),
    "force_pair_mode": (
        r'^FORCE_PAIR_MODE="\$\{FORCE_PAIR_MODE:-(.*)\}"$',
        str,
    ),
    "force_pair_floor": (
        r'^FORCE_PAIR_FLOOR="\$\{FORCE_PAIR_FLOOR:-(.*)\}"$',
        float,
    ),
    "long_dedup_s": (r'^LONG_DEDUP_S="\$\{LONG_DEDUP_S:-(.*)\}"$', float),
    "ensemble_long_members": (
        r'^ENSEMBLE_LONG_MEMBERS="\$\{ENSEMBLE_LONG_MEMBERS:-(.*)\}"$',
        int,
    ),
    "mag_model": (r'^MAG_MODEL="\$\{MAG_MODEL:-(.*)\}"$', str),
    "cls_model": (r'^CLS_MODEL="\$\{CLS_MODEL:-(.*)\}"$', str),
    "threads": (r'^THREADS="\$\{THREADS:-(.*)\}"$', int),
    "device": (r'^DEVICE="\$\{DEVICE:-(.*)\}"$', str),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_strings(value, prefix: str = ""):
    if isinstance(value, str):
        yield prefix, value
    elif isinstance(value, Mapping):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from _iter_strings(child, child_prefix)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _iter_strings(child, f"{prefix}[{index}]")


def machine_specific_strings(manifest: Mapping) -> list[str]:
    problems: list[str] = []
    for location, value in _iter_strings(manifest):
        normalized = value.replace("\\", "/")
        if re.match(r"^[A-Za-z]:/", normalized) or normalized.startswith(
            ("/home/", "/root/", "/Users/")
        ):
            problems.append(f"{location}={value!r}")
    return problems


def _literal(node: ast.AST, constants: Mapping[str, object]):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name) and node.id in constants:
        return constants[node.id]
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
    ):
        return -node.operand.value
    raise ValueError(f"无法静态解析默认值: {ast.dump(node)}")


def _python_constants(path: Path) -> dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    constants: dict[str, object] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            constants[target.id] = _literal(node.value, constants)
        except ValueError:
            continue
    return constants


def parse_serve_defaults(repo_root: Path) -> dict[str, object]:
    defaults = _python_constants(repo_root / "src" / "phasepicker" / "defaults.py")
    serve_path = repo_root / "scripts" / "serve_api.py"
    tree = ast.parse(serve_path.read_text(encoding="utf-8"), filename=str(serve_path))
    result: dict[str, object] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        option = node.args[0].value
        if option not in SERVE_ARGUMENT_TO_CONFIG:
            continue
        default_node = next(
            (keyword.value for keyword in node.keywords if keyword.arg == "default"),
            None,
        )
        if default_node is None:
            continue
        result[SERVE_ARGUMENT_TO_CONFIG[option]] = _literal(default_node, defaults)
    return result


def parse_deploy_defaults(repo_root: Path) -> dict[str, object]:
    text = (repo_root / "deploy" / "deploy_api.sh").read_text(encoding="utf-8")
    lines = text.splitlines()
    result: dict[str, object] = {}
    for key, (pattern, caster) in DEPLOY_DEFAULT_PATTERNS.items():
        match = next((re.match(pattern, line) for line in lines if re.match(pattern, line)), None)
        if match is None:
            raise ValueError(f"deploy_api.sh 缺少可解析默认值: {key}")
        result[key] = caster(match.group(1))
    return result


def parse_deploy_policy_defaults(repo_root: Path) -> dict[str, object]:
    text = (repo_root / "deploy" / "deploy_api.sh").read_text(encoding="utf-8")
    match = re.search(
        r'^ALLOW_MODEL_FALLBACK="\$\{ALLOW_MODEL_FALLBACK:-(0|1)\}"$',
        text,
        flags=re.MULTILINE,
    )
    if match is None:
        raise ValueError("deploy_api.sh 缺少可解析 ALLOW_MODEL_FALLBACK 默认值")
    return {"allow_model_fallback_default": match.group(1) == "1"}


def _same(actual, expected) -> bool:
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12)
    return actual == expected


def _inside_repo(repo_root: Path, relative_path: str) -> Path | None:
    candidate = Path(relative_path)
    if candidate.is_absolute():
        return None
    root = repo_root.resolve()
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        return None
    return resolved


def _is_git_tracked(repo_root: Path, relative_path: str) -> bool:
    completed = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", relative_path],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


def _check_asset(
    asset: Mapping,
    *,
    repo_root: Path,
    tracked: bool,
    required: bool,
    check_git: bool,
    errors: list[str],
    warnings: list[str],
) -> None:
    relative_path = str(asset.get("path", ""))
    resolved = _inside_repo(repo_root, relative_path)
    if resolved is None:
        errors.append(f"资产路径越出仓库或为绝对路径: {relative_path!r}")
        return
    if not resolved.is_file():
        message = f"资产缺失: {relative_path}"
        (errors if required else warnings).append(message)
        return
    expected_size = int(asset["size_bytes"])
    actual_size = resolved.stat().st_size
    if actual_size != expected_size:
        errors.append(
            f"资产大小不匹配 {relative_path}: {actual_size} != {expected_size}"
        )
    expected_hash = str(asset["sha256"]).lower()
    actual_hash = sha256_file(resolved)
    if actual_hash != expected_hash:
        errors.append(
            f"资产 SHA-256 不匹配 {relative_path}: {actual_hash} != {expected_hash}"
        )
    if tracked and check_git and not _is_git_tracked(repo_root, relative_path):
        errors.append(f"清单标为 tracked 但 Git 未跟踪: {relative_path}")


def _check_git_commits(
    manifest: Mapping,
    *,
    repo_root: Path,
    errors: list[str],
    warnings: list[str],
) -> None:
    if not (repo_root / ".git").exists():
        warnings.append("当前目录不是 Git 工作树，跳过提交祖先校验")
        return
    repository = manifest["repository"]
    for key in ("validated_code_commit", "deployment_record_commit", "rollback_commit"):
        commit = repository[key]
        exists = subprocess.run(
            ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
            cwd=repo_root,
            capture_output=True,
        )
        if exists.returncode != 0:
            errors.append(f"清单提交不存在: {key}={commit}")
    validated = repository["validated_code_commit"]
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", validated, "HEAD"],
        cwd=repo_root,
        capture_output=True,
    )
    if ancestor.returncode != 0:
        errors.append(f"当前 HEAD 不是已验收生产提交 {validated} 的后代")


def verify_manifest(
    manifest: Mapping,
    *,
    repo_root: Path = ROOT,
    require_external: bool = False,
    check_git: bool = True,
) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    if manifest.get("schema_version") != 1:
        errors.append("不支持的 production manifest schema_version")
    if manifest.get("status") != "deployed_and_verified":
        errors.append("production manifest 状态不是 deployed_and_verified")
    machine_paths = machine_specific_strings(manifest)
    if machine_paths:
        errors.extend(f"清单含机器绝对路径: {item}" for item in machine_paths)

    config = manifest.get("production_config", {})
    policy = manifest.get("deployment_policy", {})
    try:
        serve_defaults = parse_serve_defaults(repo_root)
        deploy_defaults = parse_deploy_defaults(repo_root)
        deploy_policy_defaults = parse_deploy_policy_defaults(repo_root)
    except (OSError, SyntaxError, ValueError) as exc:
        errors.append(f"无法解析代码/部署默认值: {exc}")
        serve_defaults = {}
        deploy_defaults = {}
        deploy_policy_defaults = {}
    for source_name, values in (
        ("serve_api", serve_defaults),
        ("deploy_api", deploy_defaults),
    ):
        for key, actual in values.items():
            if key not in config:
                errors.append(f"production_config 缺少 {key}（来自 {source_name}）")
            elif not _same(actual, config[key]):
                errors.append(
                    f"{source_name} 默认值漂移 {key}: {actual!r} != {config[key]!r}"
                )
    for key, actual in deploy_policy_defaults.items():
        if key not in policy:
            errors.append(f"deployment_policy 缺少 {key}")
        elif not _same(actual, policy[key]):
            errors.append(
                f"deploy_api 策略默认值漂移 {key}: {actual!r} != {policy[key]!r}"
            )
    expected_policy = {
        "verify_before_service_restart": True,
        "require_external_for_default_seismicxm": True,
        "allow_model_fallback_default": False,
        "emergency_fallback_environment": "ALLOW_MODEL_FALLBACK=1",
        "reject_present_but_invalid_external_asset": True,
    }
    for key, expected in expected_policy.items():
        if key not in policy:
            errors.append(f"deployment_policy 缺少 {key}")
        elif not _same(policy[key], expected):
            errors.append(
                f"deployment_policy 非预期 {key}: {policy[key]!r} != {expected!r}"
            )

    assets = manifest.get("assets", {})
    for asset in assets.get("tracked", []):
        _check_asset(
            asset,
            repo_root=repo_root,
            tracked=True,
            required=True,
            check_git=check_git,
            errors=errors,
            warnings=warnings,
        )
    for asset in assets.get("external", []):
        required = require_external and bool(asset.get("required_by_default"))
        _check_asset(
            asset,
            repo_root=repo_root,
            tracked=False,
            required=required,
            check_git=check_git,
            errors=errors,
            warnings=warnings,
        )
    if check_git:
        _check_git_commits(
            manifest,
            repo_root=repo_root,
            errors=errors,
            warnings=warnings,
        )
    return {
        "pass": not errors,
        "release_id": manifest.get("release_id"),
        "tracked_assets": len(assets.get("tracked", [])),
        "external_assets": len(assets.get("external", [])),
        "require_external": require_external,
        "errors": errors,
        "warnings": warnings,
    }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument(
        "--require-external",
        action="store_true",
        help="要求默认生产外置资产存在；部署前必须启用",
    )
    parser.add_argument("--skip-git", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    manifest_path = Path(args.manifest)
    result = verify_manifest(
        load_manifest(manifest_path),
        repo_root=ROOT,
        require_external=args.require_external,
        check_git=not args.skip_git,
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        status = "PASS" if result["pass"] else "FAIL"
        print(
            f"{status} {result['release_id']} | "
            f"tracked={result['tracked_assets']} external={result['external_assets']}"
        )
        for warning in result["warnings"]:
            print(f"WARN {warning}")
        for error in result["errors"]:
            print(f"ERROR {error}")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
