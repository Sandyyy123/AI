# Plan: Local RAG Improvement Workflow
Date: 2026-03-06
Status: Approved

## Objective
Build a WAT-based local workflow that mirrors ultrarag's Dashboard + Codebase Browser — fetch GitHub RAG repos, browse them, identify gaps vs /root/AI, generate scripts to fill those gaps, and track progress via a terminal dashboard.

## Repos to Study
- `Sandyyy123/GenAIEngineering-Cohort3` (primary)
- Additional repos discovered via analyze step

## Tools to Build (in order)

### 1. tools/fetch_github_repo.py
- Fetch repo file tree + file contents from GitHub API
- Cache to `.tmp/repos/<repo_name>/` as JSON files
- Support `--repo`, `--force` (re-fetch), `--tree-only` (no content)
- Uses GITHUB_TOKEN from .env for auth
- sys.exit(0/1), manifest write

### 2. tools/browse_repo.py
- Terminal file browser for cached repos
- Args: `--repo`, `--path` (subdirectory), `--search` (grep within files), `--file` (read specific file)
- Reads from `.tmp/repos/<repo_name>/`
- sys.exit(0/1)

### 3. tools/analyze_rag_concepts.py
- Scans cached repos for RAG techniques (chunking, embedding, retrieval, reranking, evaluation, etc.)
- Compares against /root/AI codebase
- Outputs gap report to `.tmp/rag_gap_report.json` + prints summary table
- Uses Claude (Haiku) for analysis
- sys.exit(0/1), manifest write

### 4. tools/generate_rag_script.py
- Takes a technique name from the gap report
- Reads the relevant source files from cached repo
- Uses Claude (Sonnet) to generate a Python script
- Saves to `/root/AI/tools/generated/<script_name>.py`
- Args: `--technique`, `--dry-run`
- sys.exit(0/1), manifest write

### 5. tools/rag_dashboard.py
- Terminal dashboard showing:
  - Repos fetched (name, file count, last synced)
  - RAG concepts found vs implemented (coverage %)
  - Scripts generated
  - Top 3 gaps to fix next
- Reads from .tmp/ files
- sys.exit(0)

## Workflow
- `workflows/rag_improvement.md` — SOP tying all 5 tools together

## Output Locations
- Cached repos: `.tmp/repos/`
- Gap report: `.tmp/rag_gap_report.json`
- Generated scripts: `/root/AI/tools/generated/`
- Dashboard data: reads from existing .tmp/ files

## Standards
- All tools follow WAT standards (sys.exit, manifest, validate_input)
- GITHUB_TOKEN checked via check_env.py before fetch
- Haiku for analysis, Sonnet for generation
