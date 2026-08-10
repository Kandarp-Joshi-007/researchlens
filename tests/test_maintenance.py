"""Orphan cleanup.

Regression guard: an earlier ad-hoc version of this compared relative paths
against the absolute ones stored in the database, matched nothing, and deleted
every tracked PDF along with the orphans.
"""

from pathlib import Path

from backend.core.maintenance import (
    delete_orphan_uploads, find_missing_uploads, find_orphan_uploads,
)


def _pdf(directory, name):
    path = Path(directory) / name
    path.write_bytes(b"%PDF-1.4 fake")
    return path


class TestOrphanDetection:
    def test_tracked_files_are_never_orphans(self, temp_db, tmp_path):
        tracked = _pdf(tmp_path, "keep.pdf")
        orphan = _pdf(tmp_path, "drop.pdf")
        temp_db.save_paper("keep.pdf", "T", "A", 1, filepath=str(tracked))

        orphans = find_orphan_uploads(tmp_path)
        assert orphans == [orphan]
        assert tracked.exists()

    def test_matches_regardless_of_path_form(self, temp_db, tmp_path):
        """The database may hold a non-normalised path; it still counts as tracked."""
        tracked = _pdf(tmp_path, "keep.pdf")
        awkward = str(tmp_path / "sub" / ".." / "keep.pdf")
        (tmp_path / "sub").mkdir()
        temp_db.save_paper("keep.pdf", "T", "A", 1, filepath=awkward)

        assert find_orphan_uploads(tmp_path) == []
        assert tracked.exists()

    def test_missing_directory_is_not_an_error(self, temp_db, tmp_path):
        assert find_orphan_uploads(tmp_path / "nope") == []


class TestDeletion:
    def test_dry_run_removes_nothing(self, temp_db, tmp_path):
        orphan = _pdf(tmp_path, "drop.pdf")
        result = delete_orphan_uploads(tmp_path)
        assert result["count"] == 1 and result["deleted"] is False
        assert orphan.exists(), "dry run must not delete"

    def test_explicit_delete_removes_only_orphans(self, temp_db, tmp_path):
        tracked = _pdf(tmp_path, "keep.pdf")
        orphan = _pdf(tmp_path, "drop.pdf")
        temp_db.save_paper("keep.pdf", "T", "A", 1, filepath=str(tracked))

        result = delete_orphan_uploads(tmp_path, dry_run=False)
        assert result["count"] == 1 and result["bytes"] > 0
        assert tracked.exists(), "tracked file must survive"
        assert not orphan.exists()

    def test_nothing_to_do(self, temp_db, tmp_path):
        assert delete_orphan_uploads(tmp_path, dry_run=False)["count"] == 0


class TestMissingUploads:
    def test_reports_papers_whose_file_vanished(self, temp_db, tmp_path):
        present = _pdf(tmp_path, "here.pdf")
        temp_db.save_paper("here.pdf", "T", "A", 1, filepath=str(present))
        temp_db.save_paper("gone.pdf", "T", "A", 1,
                           filepath=str(tmp_path / "gone.pdf"))

        missing = find_missing_uploads()
        assert [m["filename"] for m in missing] == ["gone.pdf"]
