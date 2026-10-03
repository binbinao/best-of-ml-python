# tests/test_history.py
import sys, pathlib, sqlite3, csv
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "scripts"))
from import_history import import_history

WEEK = "2025-10-30"

def _make_csv(tmp_path, name, rows):
    d = tmp_path / "history"
    d.mkdir(exist_ok=True)
    cols = ["", "name", "github_id", "category", "projectrank", "star_count", "last_commit_pushed_at", "description"]
    with open(d / f"{name}_projects.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(rows)
    return str(d)

def test_import_writes_files_and_rows(tmp_path):
    db = str(tmp_path / "h.db")
    rows = [[WEEK + "x", "Alpha", "org/a", "ml-frameworks", 10.0, 100, "2025-10-01", "d1"]]
    files, n = import_history(_make_csv(tmp_path, WEEK, rows), db)
    assert (files, n) == (1, 1)
    con = sqlite3.connect(db)
    assert con.execute("SELECT name, star_count FROM projects_history WHERE week=?", (WEEK,)).fetchall() == [("Alpha", 100)]

def test_import_handles_two_weeks_and_missing_name_col(tmp_path):
    db = str(tmp_path / "h.db")
    _make_csv(tmp_path, WEEK, [["x", "Alpha", "org/a", "ml-frameworks", 10.0, 100, "2025-10-01", "d1"]])
    _make_csv(tmp_path, "2025-10-23", [["x", "Beta", "org/b", "nlp", 5.0, 50, "2025-09-01", "d2"]])
    files, n = import_history(str(tmp_path / "history"), db)
    assert (files, n) == (2, 2)

from analyze import query_top, query_trend, query_recent, query_status

def test_queries_against_imported_db(tmp_path):
    db = str(tmp_path / "h.db")
    rows = [
        [WEEK + "x", "Alpha", "org/a", "ml-frameworks", 10.0, 100, "2025-10-01", "d1"],
        [WEEK + "x", "Beta", "org/b", "ml-frameworks", 9.0, 50, "2025-09-01", "d2"],
    ]
    import_history(_make_csv(tmp_path, WEEK, rows), db)
    top = query_top(db, "ml-frameworks", 2)
    assert [t[0] for t in top] == ["Alpha", "Beta"]
    trend = query_trend(db, "Alpha")
    assert trend[-1][1] == 100
    recent = query_recent(db, 5)
    assert recent[0][0] == "Alpha"
    status = query_status(db, "Beta")
    assert status["star_count"] == 50