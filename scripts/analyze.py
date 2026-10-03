#!/usr/bin/env python3
"""Learning-navigation queries over the imported history database."""
import sqlite3
import sys

DB = "data/history.db"


def _connect(db_path):
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    return con


def query_top(db_path: str, category: str, n: int = 10):
    """Top-N projects in a category by latest-week projectrank."""
    with _connect(db_path) as con:
        latest = con.execute("SELECT MAX(week) FROM projects_history").fetchone()[0]
        return con.execute(
            "SELECT name, projectrank, star_count, description FROM projects_history "
            "WHERE week = ? AND category = ? ORDER BY projectrank DESC, name LIMIT ?",
            (latest, category, n),
        ).fetchall()


def query_trend(db_path: str, name: str):
    """Weekly (week, star_count, projectrank) trajectory for a project."""
    with _connect(db_path) as con:
        return con.execute(
            "SELECT week, star_count, projectrank FROM projects_history "
            "WHERE name = ? ORDER BY week", (name,)
        ).fetchall()


def query_recent(db_path: str, n: int = 10):
    """Most recently added projects (by first-appearance week, descending)."""
    with _connect(db_path) as con:
        return con.execute(
            "SELECT name, MIN(week) AS first_week, description FROM projects_history "
            "GROUP BY name ORDER BY first_week DESC, name LIMIT ?", (n,)
        ).fetchall()


def query_status(db_path: str, name: str):
    """Latest-snapshot status fields for a project."""
    with _connect(db_path) as con:
        return con.execute(
            "SELECT * FROM projects_history WHERE name = ? "
            "ORDER BY week DESC LIMIT 1", (name,)
        ).fetchone()


def main():
    db = sys.argv[1] if len(sys.argv) > 1 else DB
    cmd = sys.argv[2] if len(sys.argv) > 2 else "top"
    args = sys.argv[3:]
    if cmd == "top":
        category, n = args[0], int(args[1]) if len(args) > 1 else 10
        for r in query_top(db, category, n):
            print(f"{r['name']}: rank={r['projectrank']:.2f} stars={r['star_count']} — {r['description']}")
    elif cmd == "trend":
        for r in query_trend(db, args[0]):
            print(f"{r['week']}: stars={r['star_count']} rank={r['projectrank']:.2f}")
    elif cmd == "recent":
        for r in query_recent(db, int(args[0]) if args else 10):
            print(f"{r['first_week']}: {r['name']} — {r['description']}")
    elif cmd == "status":
        r = query_status(db, args[0])
        for k in r.keys():
            print(f"{k}: {r[k]}")
    else:
        print(f"unknown command: {cmd}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()