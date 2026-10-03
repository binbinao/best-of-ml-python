#!/usr/bin/env python3
"""Import history/*_projects.csv snapshots into a SQLite database."""
import csv
import glob
import os
import sqlite3
import sys


def import_history(history_dir: str, db_path: str) -> tuple[int, int]:
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    con = sqlite3.connect(db_path)
    paths = sorted(glob.glob(os.path.join(history_dir, "*_projects.csv")))
    if not paths:
        con.commit()
        con.close()
        return 0, 0
    # Real history CSVs evolved schema week to week. Build the union of all
    # columns so every file can be inserted by name with NULLs for the gaps.
    all_cols: list[str] = []
    seen: set[str] = set()
    for path in paths:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
        for c in header:
            c = c or "idx"
            if c not in seen:
                seen.add(c)
                all_cols.append(c)
    col_defs = ", ".join(f"{c} NUMERIC" for c in all_cols)
    con.execute(
        f"CREATE TABLE IF NOT EXISTS projects_history (week TEXT, {col_defs}, PRIMARY KEY (week, github_id))"
    )
    col_index = {c: i for i, c in enumerate(all_cols)}
    files = rows = 0
    for path in paths:
        week = os.path.basename(path).removesuffix("_projects.csv")
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = [c or "idx" for c in next(reader)]
            present = [(c, i) for i, c in enumerate(header) if c in col_index]
            names = ", ".join(["week"] + [c for c, _ in present])
            placeholders = ", ".join("?" * (len(present) + 1))
            for rec in reader:
                values = [week] + [rec[i] for _, i in present]
                con.execute(
                    f"INSERT OR REPLACE INTO projects_history ({names}) VALUES ({placeholders})",
                    values,
                )
                rows += 1
        files += 1
    con.commit()
    con.close()
    return files, rows


if __name__ == "__main__":
    history_dir = sys.argv[1] if len(sys.argv) > 1 else "history"
    db_path = sys.argv[2] if len(sys.argv) > 2 else "data/history.db"
    files, rows = import_history(history_dir, db_path)
    print(f"imported {rows} rows from {files} weekly snapshots into {db_path}")