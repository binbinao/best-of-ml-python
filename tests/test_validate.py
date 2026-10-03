# tests/test_validate.py
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "scripts"))
from validate import validate_projects_yaml

import yaml

def _write(tmp_path, doc):
    p = tmp_path / "projects.yaml"
    p.write_text(yaml.dump(doc, sort_keys=False), encoding="utf-8")
    return str(p)

def _valid_doc():
    return {
        "configuration": {"min_stars": 300},
        "categories": [{"category": "ml-frameworks", "title": "ML", "subtitle": "s"}],
        "labels": [{"label": "pytorch", "image": "http://x/i.png", "description": "d"}],
        "projects": [
            {"name": "Alpha", "github_id": "org/repo", "category": "ml-frameworks", "labels": ["pytorch"]},
            {"name": "Beta", "github_id": "org/repo2"},  # 无 category 合法(others 兜底)
        ],
    }

def test_valid_yaml_has_no_errors(tmp_path):
    assert validate_projects_yaml(_write(tmp_path, _valid_doc())) == []

def test_missing_name_flagged(tmp_path):
    doc = _valid_doc()
    doc["projects"][0].pop("name")
    errs = validate_projects_yaml(_write(tmp_path, doc))
    assert any("missing required key 'name'" in e for e in errs)

def test_duplicate_name_flagged(tmp_path):
    doc = _valid_doc()
    doc["projects"][1]["name"] = "Alpha"
    errs = validate_projects_yaml(_write(tmp_path, doc))
    assert any("duplicate 'name'" in e for e in errs)

def test_undefined_label_flagged(tmp_path):
    doc = _valid_doc()
    doc["projects"][0]["labels"] = ["nonexistent"]
    errs = validate_projects_yaml(_write(tmp_path, doc))
    assert any("undefined label 'nonexistent'" in e for e in errs)

def test_undefined_category_flagged(tmp_path):
    doc = _valid_doc()
    doc["projects"][0]["category"] = "ghost-cat"
    errs = validate_projects_yaml(_write(tmp_path, doc))
    assert any("undefined category 'ghost-cat'" in e for e in errs)

def test_bad_github_id_flagged(tmp_path):
    doc = _valid_doc()
    doc["projects"][0]["github_id"] = "no-slash-here"
    errs = validate_projects_yaml(_write(tmp_path, doc))
    assert any("malformed 'github_id'" in e for e in errs)

def test_missing_top_level_keys_flagged(tmp_path):
    p = tmp_path / "projects.yaml"
    p.write_text("configuration: {}\n", encoding="utf-8")
    errs = validate_projects_yaml(str(p))
    assert sum("missing top-level key" in e for e in errs) == 3

def test_yaml_parse_error_reported(tmp_path):
    p = tmp_path / "projects.yaml"
    p.write_text("projects: [\n", encoding="utf-8")
    errs = validate_projects_yaml(str(p))
    assert any("YAML parse error" in e for e in errs)