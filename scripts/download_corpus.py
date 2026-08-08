"""Rebuild the document corpus from scratch.

The corpus is ~78 MB of cloned repositories, so it is gitignored rather than
committed. This script is what makes that safe: anyone who clones this repo can
reproduce the exact same corpus with one command.

    python scripts/download_corpus.py

Three git flags do the heavy lifting:

  --depth 1          only the latest commit, not years of history
  --filter=blob:none only download file contents we actually check out
  --sparse           check out nothing by default, then pick directories

Without them, `kubernetes/website` alone is over a gigabyte.

We deliberately keep each clone's .git directory. `git log` is where the
`last_updated` metadata field comes from, and that field is what makes the
"outdated document" test cases in the evaluation set possible.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# (local folder name, clone URL, [sparse-checkout paths])
SOURCES: list[tuple[str, str, list[str]]] = [
    (
        "fastapi",
        "https://github.com/fastapi/fastapi.git",
        ["docs/en/docs"],
    ),
    (
        "k8s-website",
        "https://github.com/kubernetes/website.git",
        [
            "content/en/docs/tasks",
            "content/en/docs/concepts",
            "content/en/docs/reference/kubectl",
        ],
    ),
    (
        "posthog",
        "https://github.com/PostHog/posthog.com.git",
        ["contents/handbook", "contents/docs", "contents/tutorials"],
    ),
    (
        "zulip",
        "https://github.com/zulip/zulip.git",
        ["starlight_help/src/content", "api_docs", "docs"],
    ),
]


def run(cmd: list[str]) -> None:
    """Run a command and abort loudly if it fails.

    check=True turns a non-zero exit code into an exception. Silent partial
    failure here would mean a half-downloaded corpus and evaluation numbers that
    are quietly wrong - much worse than a crash.
    """
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def download(name: str, url: str, paths: list[str], *, force: bool = False) -> None:
    target = RAW_DIR / name

    if target.exists() and not force:
        print(f"[skip] {name} already present (use --force to re-download)")
        return

    if target.exists() and force:
        print(f"[clean] removing {target}")
        # onexc handles Windows read-only files inside .git, which shutil.rmtree
        # otherwise refuses to delete.
        import shutil
        import stat

        def _force_remove(func, path, _exc):
            Path(path).chmod(stat.S_IWRITE)
            func(path)

        shutil.rmtree(target, onexc=_force_remove)

    print(f"[clone] {name}")
    run([
        "git", "clone",
        "--depth", "1",
        "--filter=blob:none",
        "--sparse",
        url,
        str(target),
    ])
    run(["git", "-C", str(target), "sparse-checkout", "set", *paths])

    # Fetch full commit history.
    #
    # --depth 1 gives one commit, which means `git log` reports the same date for
    # every file: the moment you cloned. That silently destroys the last_updated
    # metadata field, and with it the "outdated document" test cases in the
    # evaluation set. The bug is invisible - you get dates, they are just all wrong.
    #
    # --filter=blob:none keeps this affordable: we download commits and trees
    # (which is all `git log --name-only` needs) but not historical file contents.
    # Measured cost: 17s for fastapi, 108s for kubernetes/website.
    print(f"[history] {name} (needed for last_updated metadata)")
    run(["git", "-C", str(target), "fetch", "--unshallow", "--filter=blob:none"])


def main() -> int:
    force = "--force" in sys.argv
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for name, url, paths in SOURCES:
        try:
            download(name, url, paths, force=force)
        except subprocess.CalledProcessError as exc:
            print(f"[error] {name} failed: {exc}", file=sys.stderr)
            return 1

    total = sum(1 for _ in RAW_DIR.rglob("*.md")) + sum(1 for _ in RAW_DIR.rglob("*.mdx"))
    print(f"\nDone. {total} markdown/mdx documents in {RAW_DIR}")
    print("See data/CORPUS.md for licenses and what each source contributes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
