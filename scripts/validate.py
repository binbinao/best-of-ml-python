#!/usr/bin/env python3
"""Validate projects.yaml structure before the best-of generator runs."""
import re
import sys

import yaml

REQUIRED_TOP_LEVEL = ("configuration", "categories", "labels", "projects")
GITHUB_ID_RE = re.compile(r"^[^/\s]+/[^/\s]+$")


def validate_projects_yaml(path: str) -> list[str]:
    errors: list[str] = []
    try:
        with open(path, encoding="utf-8") as f:
            doc = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        return [f"{path}: YAML parse error: {exc}"]
    if not isinstance(doc, dict):
        return [f"{path}: top level is not a mapping"]
    for key in REQUIRED_TOP_LEVEL:
        if key not in doc:
            errors.append(f"{path}: missing top-level key '{key}'")
    if errors:
        return errors

    category_ids = {c.get("category") for c in doc["categories"] if isinstance(c, dict)} | {"others"}
    label_ids = {l.get("label") for l in doc["labels"] if isinstance(l, dict)}
    seen_names: dict[str, int] = {}
    for idx, proj in enumerate(doc["projects"], start=1):
        if not isinstance(proj, dict):
            errors.append(f"{path}: projects[{idx}] is not a mapping")
            continue
        name = proj.get("name")
        if not name:
            errors.append(f"{path}: project #{idx}: missing required key 'name'")
        elif name in seen_names:
            errors.append(
                f"{path}: project '{name}' (#{idx}): duplicate 'name', first seen at project #{seen_names[name]}"
            )
        else:
            seen_names[name] = idx
        github_id = proj.get("github_id")
        if not github_id or not GITHUB_ID_RE.match(str(github_id)):
            errors.append(f"{path}: project '{name or idx}': malformed 'github_id': {github_id!r} (expected 'owner/repo')")
        category = proj.get("category")
        if category is not None and category not in category_ids:
            errors.append(f"{path}: project '{name or idx}': undefined category '{category}'")
        labels = proj.get("labels")
        if labels is None:
            pass
        elif not isinstance(labels, list):
            errors.append(f"{path}: project '{name or idx}': labels must be a list (got {type(labels).__name__})")
        else:
            for label in labels:
                if label not in label_ids:
                    errors.append(f"{path}: project '{name or idx}': undefined label '{label}'")
    return errors


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "projects.yaml"
    errors = validate_projects_yaml(path)
    for err in errors:
        print(err, file=sys.stderr)
    print(f"{path}: {'OK' if not errors else f'{len(errors)} error(s)'}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())