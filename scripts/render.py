#!/usr/bin/env python3
"""Render the book and check that the committed freeze cache is in sync.

_quarto.yml sets `execute: freeze: auto`, so quarto only re-executes a
chapter's Python (and mermaid) chunks when its source has changed since the
result under _freeze/ was committed; otherwise it reuses the frozen output.
Unlike a bare CI runner, .github/workflows/publish.yml installs the full
scientific stack (conda: numpy, scipy, scikit-learn, evomsa, jupyter,
IngeoML, pandas, seaborn, nltk, compstats, umap-learn; pip: jax, optax)
before running `quarto publish gh-pages .`, so CI can always fall back to
executing a chapter itself — a stale cache does not break the build there.

What a stale or incomplete cache does cost:
  - CI silently re-executing chapters whose freeze entry you forgot to
    refresh, which is slow (several long-running fits across the book) and
    can render with different randomness than what you reviewed locally.
  - A chapter whose source is unchanged (so quarto trusts the cache) but
    whose figure files were deleted from _freeze/ some other way — that
    publishes as a broken image, and nothing else catches it.

This script renders the book, then checks every chapter's frozen result the
way quarto's freeze mechanism reads it: source hash must match, and every
figure the frozen markdown points at must exist under _freeze/.

Usage, from anywhere in the repo:

    python scripts/render.py                              # deps, render, verify
    python scripts/render.py --verify-only                # check the cache only
    python scripts/render.py capitulos/01Introduccion.qmd
    python scripts/render.py --force capitulos/01Introduccion.qmd

quarto itself is not installed by this script: it comes from the
devcontainer in .devcontainer/ (the rocker-org quarto-cli feature), so run
this inside that container, or install quarto some other way first.

Rendering the whole book executes every chapter's code (model fits, plots,
UMAP embeddings, etc.), so expect several minutes. Both _book/ and the
downloaded datasets sklearn/nltk cache locally are gitignored; the
artifacts to commit are _freeze plus any chapter whose stored figures
changed, and the script prints them at the end.
"""

import argparse
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FREEZE = REPO / "_freeze"
QUARTO_YML = REPO / "_quarto.yml"
WORKFLOW = REPO / ".github" / "workflows" / "publish.yml"
REQUIREMENTS = REPO / "requirements.txt"

# What the chapters import, as {module: pip requirement}, checked before
# rendering so a missing package fails fast with a clear fix instead of a
# mid-render ImportError. requirements.txt is what actually gets installed
# (matching .devcontainer/postCreate.sh) since these packages pull in
# numpy/scipy/scikit-learn/matplotlib transitively.
CHECK_MODULES = {
    "pandas": "pandas",
    "seaborn": "seaborn",
    "IngeoML": "IngeoML",
    "EvoMSA": "EvoMSA",
    "wordcloud": "wordcloud",
    "jupyterlab": "jupyterlab",
    "umap": "umap-learn",
    "nltk": "nltk",
    "CompStats": "compstats",
    "jax": "jax",
    "optax": "optax",
}

# quarto's freeze result file per format configured in _quarto.yml.
RESULT_FILES = {"html": "html.json", "pdf": "tex.json"}


def log(message: str) -> None:
    print(f"[render] {message}", flush=True)


def pages() -> list[Path]:
    """The book's source files, in the order _quarto.yml lists them."""
    text = QUARTO_YML.read_text(encoding="utf-8")
    entries = re.findall(r"^\s*-\s+(\S+\.qmd)\s*$", text, re.MULTILINE)
    return [Path(entry) for entry in dict.fromkeys(entries)]


def is_executable(page: Path) -> bool:
    """Whether `page` has any code chunk (python, mermaid, ...) to freeze.

    A few pages (index.qmd, capitulos/17Referencias.qmd) are plain markdown
    and never get a _freeze/ entry at all, so they must be skipped rather
    than flagged as missing.
    """
    text = (REPO / page).read_text(encoding="utf-8")
    return re.search(r"^```\{", text, re.MULTILINE) is not None


def freeze_dir(page: Path) -> Path:
    """Where quarto keeps `page`'s frozen execution results and figures."""
    return FREEZE / page.with_suffix("")


def quarto_version() -> str:
    """The quarto on PATH, or exit telling the caller where to get one."""
    try:
        out = subprocess.run(
            ["quarto", "--version"], capture_output=True, text=True, check=True
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        sys.exit(
            "[render] quarto not found on PATH. Open this repo in the devcontainer "
            "under .devcontainer/, which installs it, or install quarto separately."
        )
    return out.stdout.strip()


def ci_quarto_version() -> str | None:
    """The version publish.yml pins for quarto-actions/setup, if any."""
    if not WORKFLOW.exists():
        return None
    text = WORKFLOW.read_text(encoding="utf-8")
    match = re.search(
        r"quarto-actions/setup@[^\n]*\n(?:\s*\n)*\s*with:[^\n]*\n\s*version:\s*"
        r"[\"']?([\d.]+)",
        text,
    )
    return match.group(1) if match else None


def ensure_dependencies() -> None:
    """Install requirements.txt if anything the chapters import is missing."""
    missing = [
        module for module in CHECK_MODULES if importlib.util.find_spec(module) is None
    ]
    if not missing:
        log("rendering dependencies already installed")
        return
    log(f"missing {', '.join(missing)} — installing {REQUIREMENTS.name}")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS)], check=True
    )


def render(targets: list[Path], force: bool) -> None:
    """Run quarto over `targets` (the whole book when empty)."""
    if force:
        for page in targets or pages():
            stale = freeze_dir(page)
            if stale.exists():
                log(f"dropping frozen result for {page}")
                shutil.rmtree(stale)

    command = ["quarto", "render", *(str(t) for t in targets)]
    log(f"{' '.join(command)}  (cwd: {REPO})")
    result = subprocess.run(command, cwd=REPO)
    if result.returncode != 0:
        sys.exit(f"[render] quarto render failed with exit code {result.returncode}")


def verify() -> list[str]:
    """Check every chapter's frozen result the way quarto's freeze will read it."""
    problems = []
    for page in pages():
        if not is_executable(page):
            continue

        source = REPO / page
        digest = hashlib.md5(source.read_bytes()).hexdigest()
        stem_prefix = f"{page.stem}_files"

        for fmt, filename in RESULT_FILES.items():
            result_file = freeze_dir(page) / "execute-results" / filename
            if not result_file.exists():
                problems.append(
                    f"{page} [{fmt}]: no frozen result at "
                    f"{result_file.relative_to(REPO)} — quarto would execute this "
                    f"chapter. Render it (this script, without --verify-only)."
                )
                continue

            frozen = json.loads(result_file.read_text(encoding="utf-8"))
            if digest != frozen["hash"]:
                problems.append(
                    f"{page} [{fmt}]: stale freeze — md5(source)={digest} but the "
                    f"frozen hash is {frozen['hash']}. Re-render it, and do not edit "
                    f"the chapter afterwards without rendering again."
                )
                continue

            result = frozen["result"]
            markdown = result["markdown"]
            missing_assets = []
            for supporting in result.get("supporting", []):
                referenced = set(re.findall(rf"{re.escape(supporting)}/[\w./-]+", markdown))
                for reference in sorted(referenced):
                    relative = Path(reference).relative_to(stem_prefix)
                    asset = freeze_dir(page) / relative
                    if not asset.exists():
                        missing_assets.append(
                            f"{page} [{fmt}]: frozen markdown references {reference} "
                            f"but {asset.relative_to(REPO)} is missing."
                        )

            if missing_assets:
                problems.extend(missing_assets)
            else:
                log(f"{page} [{fmt}]: freeze OK (hash {digest[:12]})")

    # A frozen result only helps once it is committed: CI checks the repo
    # out and renders from that, so anything still untracked here is
    # invisible there.
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "--", "_freeze"],
        cwd=REPO,
        capture_output=True,
        text=True,
    ).stdout.split()
    for path in untracked:
        problems.append(f"{path} is untracked — git add it, or CI will not see it.")
    return problems


def report_artifacts() -> None:
    """Show what the render touched, so the caller knows what to commit."""
    paths = ["_freeze", *(str(page) for page in pages())]
    changed = subprocess.run(
        ["git", "status", "--porcelain", "--", *paths],
        cwd=REPO,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not changed:
        log("nothing changed — the committed cache already matches the sources")
        return
    log("commit these together, so the chapters and their freeze hashes stay in step:")
    for line in changed.splitlines():
        print(f"    {line}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "targets",
        nargs="*",
        help="chapters to render, relative to the repo root (default: the whole book)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="skip rendering; only check that the committed freeze cache is usable",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="discard the frozen results of the selected pages so they re-execute",
    )
    parser.add_argument(
        "--skip-deps",
        action="store_true",
        help="do not install missing Python dependencies before rendering",
    )
    args = parser.parse_args()

    targets = [Path(t) for t in args.targets]
    for target in targets:
        if not (REPO / target).exists():
            sys.exit(f"[render] no such page: {target}")

    if not args.verify_only:
        local = quarto_version()
        pinned = ci_quarto_version()
        log(f"quarto {local} (CI pins {pinned or 'nothing'})")
        if pinned and local != pinned:
            log(
                f"note: CI renders with quarto {pinned}. The freeze hash is a plain md5 "
                f"of the source, so a cache written here is still valid there, but "
                f"pin the devcontainer's quarto-cli feature to {pinned} if the rendered "
                f"HTML/PDF ever diverges."
            )
        if not args.skip_deps:
            ensure_dependencies()
        render(targets, args.force)

    problems = verify()
    if problems:
        log(f"{len(problems)} problem(s):")
        for problem in problems:
            print(f"    - {problem}")
        return 1

    log("every chapter's frozen result matches its source")
    if not args.verify_only:
        report_artifacts()
    return 0


if __name__ == "__main__":
    sys.exit(main())
