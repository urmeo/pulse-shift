"""Record inputs, source versions, and runtime for generated artifacts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from . import config


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact_provenance(panel_path: Path | None = None) -> dict:
    repository_root = config.ROOT.parent
    source_patterns = ("pulseshift/*.py", "scripts/*.py", "templates/*.html")
    source_files = sorted(
        path for pattern in source_patterns for path in config.ROOT.glob(pattern)
    )
    source_hashes = {
        str(path.relative_to(repository_root)): file_sha256(path)
        for path in source_files
    }
    source_digest = hashlib.sha256()
    for name, sha256 in source_hashes.items():
        source_digest.update(f"{name}\0{sha256}\n".encode())
    source_commit, source_dirty = None, None
    try:
        source_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            [
                "git",
                "status",
                "--porcelain",
                "--",
                *(f"{config.ROOT.name}/{pattern}" for pattern in source_patterns),
            ],
            cwd=repository_root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        source_dirty = bool(status.strip())
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    package_versions: dict[str, str | None] = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib"):
        try:
            package_versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            package_versions[name] = None
    panel_path = panel_path or config.PROCESSED / "panel.csv.gz"
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": source_commit,
        "source_dirty": source_dirty,
        "source_sha256": source_digest.hexdigest(),
        "source_files_sha256": source_hashes,
        "panel_sha256": file_sha256(panel_path),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": package_versions,
        "label": {
            "ratio": config.SUPPRESSION_RATIO,
            "expected_floor": config.EXPECTED_FLOOR,
            "climatology": "training-year season × daytype × hour median with training-year volume level",
            "train_years": list(config.TRAIN_YEARS),
            "test_year": config.TEST_YEAR,
        },
        "air_quality": {
            "source": "CAMS Global Atmospheric Composition Forecasts via Open-Meteo",
            "native_resolution": "approximately 45 km / 3-hourly; hourly API values",
            "fallback": "EPA daily AQI where CAMS hourly AQI is missing",
        },
        "safety_rails": {
            "heat_unsafe_f": config.HEAT_UNSAFE_F,
            "aqi_unsafe": config.AQI_UNSAFE,
        },
    }
