# best-of-ai-python 改造实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 best-of-ml-python 改造为 AI 时代 Python ML 学习清单:修复数据缺陷、新增 5 类目 + 策展批次、建立校验层、现代化 CI、激活 history 数据为 SQLite 查询层。

**Architecture:** 数据仓库三阶段串行改造。阶段一以 `scripts/validate.py` 为安全网,先修 6 处已知缺陷再扩充数据;阶段二只改 workflow 与 gitattributes,保持与上游模板最小 diff;阶段三用标准库 sqlite3 把 history CSV 导入单文件库并暴露 4 个导航查询。

**Tech Stack:** Python 3.13(本机)/ 3.10+(CI 兼容),PyYAML(唯一第三方依赖),pytest,标准库 sqlite3/GitHub Actions `gh` CLI。

**Spec:** `docs/superpowers/specs/2026-10-03-ai-era-transformation-design.md`

## Global Constraints

- 仓库无构建系统:所有代码落在 `scripts/`(标准库 + PyYAML)与 `tests/`(pytest);不得引入其他第三方依赖。
- `README.md`、`latest-changes.md`、`history/` 为生成物,任何任务不得手改(spec §2 非目标)。
- `projects.yaml` 编辑保持上游 YAML 风格:双空格缩进列表项、`name:` 首键、标签用内联 `labels: ["x"]`。
- 提交粒度:每任务一提交,消息格式 `fix:`/`feat:`/`test:`/`chore:` 前缀。
- 校验验收基线:修复前快照 ≥6 错、修复后 0 错(spec §3.5)。
- pytest 全部测试在任何任务结束时必须绿;测试文件放 `tests/`,命名 `test_<脚本名>.py`。

---

### Task 1: 校验脚本 validate.py(TDD)

**Files:**
- Create: `scripts/validate.py`
- Create: `tests/test_validate.py`
- Modify: `.gitignore`(追加 `__pycache__/`,若模板未含)

**Interfaces:**
- Consumes: `projects.yaml`(只读)
- Produces: `validate_projects_yaml(path: str) -> list[str]`,返回错误字符串列表(空列表 = 有效);`main()` 打印 `projects.yaml:<近似行号>: <描述>` 并按错误数退出非 0。后续 Task 2/3/8 依赖此函数名与签名;CI(Task 8)以退出码为门禁。

- [ ] **Step 1: 写失败测试**

```python
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
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_validate.py -v`
Expected: 全部 FAIL/ERROR(`ModuleNotFoundError: No module named 'validate'`)

- [ ] **Step 3: 最小实现**

```python
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

    category_ids = {c.get("category") for c in doc["categories"] if isinstance(c, dict)}
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
        for label in proj.get("labels") or []:
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
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/test_validate.py -v`
Expected: 8 PASS

- [ ] **Step 5: 对真实仓库跑基线验收**

Run: `python scripts/validate.py`
Expected: **恰好 6 个错误** — 1× missing name(neurolink,project #920 前后)、2× undefined label 'spacy'、3× undefined label 'huggingface'。若数量不符,停下核对 spec §3.1 表格再继续。

- [ ] **Step 6: 提交**

```bash
git add scripts/validate.py tests/test_validate.py
git commit -m "feat: add projects.yaml structural validator"
```

---

### Task 2: 修复 6 处已知数据缺陷

**Files:**
- Modify: `projects.yaml`(labels 块 ~124-151 行、561/4106/4300/4341/4353 行、4668 行)

**Interfaces:**
- Consumes: Task 1 的 `scripts/validate.py`(验收门禁)
- Produces: 零错误的 `projects.yaml`;新增标签定义 `spacy`、`huggingface`(Task 3 的策展批次会引用它们)

- [ ] **Step 1: 在 labels 块末尾(jax 定义之后)追加两个标签定义**

```yaml
  - label: "spacy"
    image: "https://spacy.io/favicon.ico"
    description: "spaCy related project"
  - label: "huggingface"
    image: "https://huggingface.co/favicon.ico"
    description: "Hugging Face related project"
```

- [ ] **Step 2: 给 neurolink 条目补 name**

定位 `projects.yaml:4668` 条目 `- github_id: juspay/neurolink`,在行前插入 `name:` 键,改为:

```yaml
  - name: neurolink
    github_id: juspay/neurolink
    category: ml-frameworks
    description: Enterprise-grade LLM integration framework for building production-ready AI applications with built-in hallucination prevention, RAG, and MCP support
```

- [ ] **Step 3: 验收**

Run: `python scripts/validate.py && python -m pytest tests/test_validate.py -q`
Expected: `projects.yaml: OK`;8 tests PASS

- [ ] **Step 4: 提交**

```bash
git add projects.yaml
git commit -m "fix: add missing neurolink name, define spacy/huggingface labels"
```

---

### Task 3: 新增 5 类目 + 4 新标签定义

**Files:**
- Modify: `projects.yaml`(categories 块 ~120-122 行 `chinese-nlp` 定义之后、labels 块追加 4 定义)

**Interfaces:**
- Consumes: Task 2 已定义 spacy/huggingface 标签
- Produces: 类目 ID `llm-frameworks`、`agent-frameworks`、`rag-vectordb`、`inference-serving`、`model-hub`;标签 ID `llm`、`agent`、`rag`、`inference`。Task 4 的项目条目引用这些 ID。

- [ ] **Step 1: 在 categories 块的 `chinese-nlp`(ignore: True)条目之后插入 5 类目**

```yaml
  - category: "llm-frameworks"
    title: "LLM Frameworks & Toolkits"
    subtitle: "Libraries for accessing, fine-tuning, quantizing, and aligning large language models."
  - category: "agent-frameworks"
    title: "Agent & Orchestration Frameworks"
    subtitle: "Frameworks for building LLM agents: tool calling, planning, and multi-agent orchestration."
  - category: "rag-vectordb"
    title: "RAG & Vector Databases"
    subtitle: "Retrieval-augmented generation pipelines and vector stores for semantic search."
  - category: "inference-serving"
    title: "LLM Inference & Serving"
    subtitle: "High-throughput LLM inference engines and serving infrastructure."
  - category: "model-hub"
    title: "Model Hubs & Registries"
    subtitle: "Model hosting, weight distribution, and registry services."
```

- [ ] **Step 2: 在 labels 块末尾(huggingface 定义之后)追加 4 标签**

```yaml
  - label: "llm"
    image: "https://img.shields.io/badge/llm-FF6F00"
    description: "LLM related project"
  - label: "agent"
    image: "https://img.shields.io/badge/agent-3F51B5"
    description: "Agent related project"
  - label: "rag"
    image: "https://img.shields.io/badge/rag-00897B"
    description: "RAG related project"
  - label: "inference"
    image: "https://img.shields.io/badge/inference-6A1B9A"
    description: "Inference serving related project"
```

- [ ] **Step 3: 验收**

Run: `python scripts/validate.py`
Expected: OK(新类目/标签尚无项目引用,合法)

- [ ] **Step 4: 提交**

```bash
git add projects.yaml
git commit -m "feat: add LLM/agent/RAG/inference/model-hub categories and labels"
```

---

### Task 4: 分类目门槛验证(spike,结论写回 spec)

**Files:**
- Modify: `docs/superpowers/specs/2026-10-03-ai-era-transformation-design.md`(§3.4 写入验证结论)

**Interfaces:**
- Consumes: 上游 best-of-generator 文档/源码(`https://github.com/best-of-lists/best-of-generator`)
- Produces: 结论 A(per-category 支持,给出确切语法)或结论 B(不支持,全局 300 维持)。Task 5 的准入门槛依赖此结论。

- [ ] **Step 1: 检查上游生成器对 per-category min_stars 的支持**

用 web 检索/读上游仓库 README 与源码(`best_of/generators/loader.py`、`projects.py` 等):搜索 `min_stars` 出现处,确认它是仅读 `configuration:` 还是也读 `categories:` 条目。记录确切证据(文件+行为)。

- [ ] **Step 2: 把结论写入 spec §3.4**

结论 A 示例文案:`已验证:best-of-generator <版本> 在 <文件> 支持 categories[].min_stars,语法 <代码>。阶段一为 5 个新类目设置 min_stars: 100。`
结论 B 示例文案:`已验证:best-of-generator 仅在 configuration 块读 min_stars(<证据>)。维持全局 300,差异化门槛转为上游 patch 待办。`

- [ ] **Step 3: 提交**

```bash
git add docs/superpowers/specs/2026-10-03-ai-era-transformation-design.md
git commit -m "docs: record per-category min_stars verification result"
```

---

### Task 5: 策展批次(38 候选 → 查重 → 用户审核 → 入库)

**Files:**
- Create: `docs/superpowers/plans/2026-10-03-curation-batch.md`(审核清单,用户逐条勾选)
- Modify: `projects.yaml`(projects 列表末尾追加审核通过的条目)

**Interfaces:**
- Consumes: Task 3 的类目/标签 ID;Task 4 的门槛结论
- Produces: 批次入库后的 `projects.yaml`(仍须过 validate)

- [ ] **Step 1: 生成审核清单文件**

对 spec §3.3 的 38 个候选,逐个:①查重(`grep 'github_id: <owner/repo>' projects.yaml`,已存在则标注 SKIP 并归入"已在库"备注);②核实当前真实星标(用 GitHub 公开数据,无 API 也可从 README 生成物或 web 检索估算,标注数据来源与日期);③按 Task 4 结论过滤门槛;④写出完整拟入库 YAML 条目。清单按类目分组,每条目带复选框。

- [ ] **Step 2: 提交清单并请用户审核**

提交 `docs/superpowers/plans/2026-10-03-curation-batch.md`,在对话中请用户逐条勾选或整体批准。**未获批准前不入库任何条目。**

- [ ] **Step 3: 用户批准后,把通过的条目追加到 projects 列表末尾**

追加位置:`projects.yaml` 末条(neurolink)之后。条目格式对齐上游(`name`/`github_id`/`pypi_id`/`conda_id`(若有)/`category`/`labels`)。

- [ ] **Step 4: 验收**

Run: `python scripts/validate.py`
Expected: OK

- [ ] **Step 5: 提交**

```bash
git add projects.yaml docs/superpowers/plans/2026-10-03-curation-batch.md
git commit -m "feat: add curated AI-era project batch"
```

---

### Task 6: 阶段二 — workflow 前置校验 + 工具链现代化

**Files:**
- Modify: `.github/workflows/update-best-of-list.yml`
- Modify: `.github/.gitattributes`

**Interfaces:**
- Consumes: Task 1 的 `scripts/validate.py`(CI 门禁);`gh` CLI(ubuntu-latest 预装)
- Produces: 更新后的 workflow;`.gitattributes` 生成物标记

- [ ] **Step 1: 在 workflow 中 checkout `update/<VERSION>` 分支的 step 之后、`update-best-of-list` step 之前插入校验 step**

```yaml
      - name: validate-projects-yaml
        run: |
          pip install pyyaml
          python scripts/validate.py
```

- [ ] **Step 2: 替换 create-pull-request step(移除 hub)**

删除整个 `create-pull-request` step(curl 装 hub + `bin/hub pull-request ...` + `rm bin/hub`),替换为:

```yaml
      - name: create-pull-request
        shell: bash
        run: |
          set -e
          gh pr create -b "${{ env.DEFAULT_BRANCH }}" -h "${{ env.BRANCH_PREFIX }}${{ env.VERSION }}" \
            --title "Best-of update: ${{ env.VERSION }}" \
            --body "To finish this update: Select \`Merge pull request\` below and \`Confirm merge\`. Also, make sure to publish the created draft release in the [releases section](../releases) as well." || true
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

- [ ] **Step 3: 替换 create-release step(移除 actions/create-release@v1)**

删除整个 `create-release` step,替换为:

```yaml
      - name: create-release
        shell: bash
        run: |
          gh release create "${{ env.VERSION }}" latest-changes.md \
            --draft --title "Update: ${{ env.VERSION }}" || true
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

- [ ] **Step 4: .github/.gitattributes 追加生成物标记**

```
* linguist-language=Python
README.md linguist-generated=true
latest-changes.md linguist-generated=true
history/** linguist-generated=true
```

- [ ] **Step 5: 验收**

Run: `python -c "import yaml; yaml.safe_load(open('.github/workflows/update-best-of-list.yml'))" && python scripts/validate.py`
Expected: 无输出(语法 OK)+ validator OK。人工核对 workflow diff:校验 step 在生成前、hub/create-release@v1 已无引用。

- [ ] **Step 6: 提交**

```bash
git add .github/workflows/update-best-of-list.yml .github/.gitattributes
git commit -m "chore: validate before generation; replace hub and create-release with gh CLI"
```

---

### Task 7: 阶段三 — history 导入 + 查询脚本(TDD)

**Files:**
- Create: `scripts/import_history.py`
- Create: `scripts/analyze.py`
- Create: `tests/test_history.py`
- Modify: `.gitignore`(追加 `data/`)

**Interfaces:**
- Consumes: `history/*_projects.csv`(只读)
- Produces: `import_history(history_dir: str, db_path: str) -> tuple[int, int]`(写入文件数,写入行数);表 `projects_history(week TEXT, ...CSV 列, PRIMARY KEY (week, github_id))`;`analyze.py` CLI 四命令:`top <category> <N>`、`trend <name>`、`recent <N>`、`status <name>`。Task 8 的 AGENTS.md 更新引用这些命令。

- [ ] **Step 1: 写失败测试(用固定小 CSV fixture)**

```python
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
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_history.py -v`
Expected: FAIL/ERROR(No module named import_history)

- [ ] **Step 3: 实现 import_history.py**

```python
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
    files = rows = 0
    for path in sorted(glob.glob(os.path.join(history_dir, "*_projects.csv"))):
        week = os.path.basename(path).removesuffix("_projects.csv")
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            cols = [c or "idx" for c in header]
            con.execute(
                f"CREATE TABLE IF NOT EXISTS projects_history ({', '.join(cols)}, PRIMARY KEY (week, github_id))"
            ) if files == 0 else None
            placeholders = ", ".join("?" * (len(cols) + 1))
            names = ", ".join(["week"] + cols)
            for rec in reader:
                con.execute(
                    f"INSERT OR REPLACE INTO projects_history ({names}) VALUES ({placeholders})",
                    [week] + rec,
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
```

注意:真实 CSV 首列名为空(索引列),fixture 已用 `"idx"` 占位逻辑处理——实现里 `c or "idx"` 即此用途。

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/test_history.py -v`
Expected: 2 PASS

- [ ] **Step 5: 写 analyze.py 的失败测试(追加到 tests/test_history.py)**

```python
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
```

- [ ] **Step 6: 运行确认失败 → 实现 analyze.py → 运行确认通过**

```python
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
```

Run: `python -m pytest tests/test_history.py -v`
Expected: 3 PASS

- [ ] **Step 7: .gitignore 追加 `data/`,全量导入冒烟**

Run: `echo "data/" >> .gitignore && python scripts/import_history.py && python scripts/analyze.py data/history.db top ml-frameworks 3`
Expected: 导入输出 `imported ~920 rows from 250+ snapshots` 量级;top 查询打印 3 行真实项目。人工核对结果合理(如 Tensorflow/PyTorch 居首)。

- [ ] **Step 8: 提交**

```bash
git add scripts/import_history.py scripts/analyze.py tests/test_history.py .gitignore
git commit -m "feat: SQLite history import and learning-navigation queries"
```

---

### Task 8: AGENTS.md 更新(收尾,含所有阶段)

**Files:**
- Modify: `AGENTS.md`

**Interfaces:**
- Consumes: Task 1/6/7 全部交付物
- Produces: 与仓库现状一致的 AI 助手指南

- [ ] **Step 1: 更新 AGENTS.md 四处**

1. **Development Commands**:删除「无本地验证命令」段落,替换为:`python scripts/validate.py`(校验)、`python scripts/import_history.py && python scripts/analyze.py`(数据查询)、`python -m pytest tests/ -v`(测试)。
2. **Testing & QA**:删除「无校验」表述,登记:validate.py 已挂 CI 生成前门禁;pytest 套件存在(`tests/test_validate.py`、`tests/test_history.py`);校验规则清单。
3. **Code Conventions**:类目表追加 5 新类目;标签清单更新为 13 个(spacy、huggingface、llm、agent、rag、inference 加入原有 9 个)。
4. **重要警示保留**:README.md/latest-changes.md/history/ 为生成物、勿手改——不变。

- [ ] **Step 2: 验收**

Run: `python scripts/validate.py && python -m pytest tests/ -q`
Expected: OK + 全部 PASS

- [ ] **Step 3: 提交**

```bash
git add AGENTS.md
git commit -m "docs: update AGENTS.md for validator, queries, and new taxonomy"
```

---

## Self-Review 记录

- Spec 覆盖:§3.1→Task 2;§3.2→Task 3;§3.3→Task 5;§3.4→Task 4;§3.5→Task 1;§4→Task 6;§5→Task 7;§6 验证→各任务验收步;AGENTS.md 更新→Task 8。无缺口。
- 占位符扫描:候选池的星标核实是流程性描述(查重/核实即执行内容),非占位;Task 5 入库条目依赖用户审核,审核清单本身就是交付物。其余步骤均含完整代码/命令。
- 类型一致性:`validate_projects_yaml(path)->list[str]`、`import_history(dir, db)->tuple[int,int]`、`query_top/trend/recent/status` 签名在 Task 1/7 定义并被 Task 6/8 引用一致。
