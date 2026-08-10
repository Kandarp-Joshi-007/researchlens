"""Housekeeping for the uploads directory and vector store.

Deleting papers can leave files behind, and analyses that fail part-way can
leave embeddings without a paper. Both cleanups compare *resolved* paths: the
database stores absolute paths, so comparing against relative ones matches
nothing and would delete every file.
"""

import logging
from pathlib import Path
from typing import List

from .database import get_conn

log = logging.getLogger(__name__)


def _tracked_paths() -> set:
    """Absolute, resolved paths of every PDF still referenced by a paper row."""
    with get_conn() as conn:
        rows = conn.execute("SELECT filepath FROM papers WHERE filepath IS NOT NULL")
        paths = set()
        for row in rows:
            try:
                paths.add(Path(row["filepath"]).resolve())
            except (OSError, ValueError):
                continue
    return paths


def find_orphan_uploads(upload_dir) -> List[Path]:
    """PDFs on disk that no paper row points at."""
    upload_dir = Path(upload_dir)
    if not upload_dir.exists():
        return []
    tracked = _tracked_paths()
    orphans = []
    for path in upload_dir.glob("*.pdf"):
        try:
            resolved = path.resolve()
        except (OSError, ValueError):
            continue
        if resolved not in tracked:
            orphans.append(path)
    return sorted(orphans)


def delete_orphan_uploads(upload_dir, dry_run: bool = True) -> dict:
    """Report, and optionally remove, untracked PDFs.

    Defaults to a dry run: this deletes user data, so the caller has to ask
    for it explicitly.
    """
    orphans = find_orphan_uploads(upload_dir)
    freed = 0
    for path in orphans:
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if not dry_run:
            try:
                path.unlink()
            except OSError as exc:
                log.warning("Could not delete %s: %s", path, exc)
                continue
        freed += size

    return {
        "files": [str(p) for p in orphans],
        "count": len(orphans),
        "bytes": freed,
        "deleted": not dry_run,
    }


def find_missing_uploads() -> List[dict]:
    """Papers whose stored PDF is no longer on disk (re-analysis will fail)."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, filename, filepath FROM papers WHERE filepath IS NOT NULL"
        ).fetchall()
    return [dict(r) for r in rows if not Path(r["filepath"]).exists()]
