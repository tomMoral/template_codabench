"""Build the Codabench bundle for the competition.

The bundle ships the competition config, the pages, the sample submission, the
ingestion/scoring programs, and the phase directory as ``main_phase`` (just the
phase ``config.yaml`` + reference placeholder). The benchopt **benchmark** is
baked into the Docker image (``/compet/benchmark``), not the bundle; the
**data** is staged onto the compute worker at ``$BENCHOPT_DATA_HOME``
(``/app/data``), which keeps the sealed private split out of the public
artifact.

Usage
-----
    python tools/create_bundle.py
"""

import os
import zipfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

# Skip benchopt run artefacts / caches / staged data when zipping.
_SKIP_PARTS = ("outputs", "__cache__", "__pycache__", ".pytest_cache", "data")


def _walk(src):
    """Every file under *src*, following symlinks (rglob does not descend
    into a symlinked directory — it would silently drop shared code)."""
    for parent, _dirs, files in os.walk(src, followlinks=True):
        for name in files:
            yield Path(parent) / name


def _add_file(bundle, src, arcname):
    print(arcname)
    bundle.write(src, arcname)


def _add_dir(bundle, src, arc_prefix):
    assert src.exists(), f"{src} does not exist while it should."
    for f in sorted(_walk(src)):
        if not f.is_file():
            continue
        if f.name.startswith(".") or f.name.endswith(".pyc"):
            continue
        if any(part in _SKIP_PARTS for part in f.relative_to(src).parts):
            continue
        _add_file(bundle, f, str(Path(arc_prefix) / f.relative_to(src)))


def build_bundle():
    out = ROOT_DIR / "bundle.zip"
    with zipfile.ZipFile(out, mode="w") as bundle:
        _add_file(bundle, ROOT_DIR / "competition.yaml", "competition.yaml")
        _add_file(bundle, ROOT_DIR / "logo.png", "logo.png")

        _add_dir(bundle, ROOT_DIR / "pages", "pages")
        _add_dir(bundle, ROOT_DIR / "ingestion_program", "ingestion_program")
        _add_dir(bundle, ROOT_DIR / "scoring_program", "scoring_program")
        _add_dir(bundle, ROOT_DIR / "solution", "solution")

        _add_file(bundle, ROOT_DIR / "phases/main/input_data/config.yaml",
                  "main_phase/input_data/config.yaml")
        _add_dir(bundle, ROOT_DIR / "phases/reference_data",
                 "main_phase/reference_data")
    print(f"-> {out.name}")


if __name__ == "__main__":
    build_bundle()
