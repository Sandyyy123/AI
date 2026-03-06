# Claude Code YouTube Learnings

Tracks what was studied, what was implemented, and what to act on.
Add new videos to the NotebookLM notebook: "Ray Amjad - Top 0.01% Claude Code Guide"
Notebook ID: 7022b4a2-54ef-499d-9630-dbdcee7adc69

---

## Videos Studied

| # | Title | URL | Source ID (NotebookLM) |
|---|-------|-----|------------------------|
| 1 | The Top 0.01% User's Guide to Claude Code — Ray Amjad | https://www.youtube.com/watch?v=AzmnaoVP8sk | 61cd2dae-dba5-464b-93bf-6331a5be094b |
| 2 | CLAUDE CODE FULL COURSE 4 HOURS: Build & Sell (2026) | https://www.youtube.com/watch?v=6-D3fg3JUL4 | 6acdcb73-9b7b-40ab-aad4-877127de331a |
| 3 | CLAUDE CODE FULL COURSE 4 HOURS: Build & Sell (2026) | https://www.youtube.com/watch?v=QoQBzR1NIqI | 4bc45935-fd1b-4879-9ca9-963b20595d9c |
| 4 | Claude Code Skills Are Broken (Beginner to Pro) | https://www.youtube.com/watch?v=zKBPwDpBfhs | 161d022e-96bb-4997-9e9c-31cdcc5c1306 |
| 5 | Claude Cowork Is the First AI That Feels Like a Real Employee | https://www.youtube.com/watch?v=56IeB1CZEYY | 757b6e6c-2441-4219-b7e3-2d97064053fd |
| 6 | Claude Can Now Run 24/7 Without You (Scheduled Tasks) | https://www.youtube.com/watch?v=A-xGGE3QQIM | 219f9f8d-2727-44b6-a6d1-323e5ce98173 |
| 7 | How To Use Claude Code FREE Forever (Ollama Setup) | https://www.youtube.com/watch?v=gqYyZuO34x0 | a416c1cf-f3bc-4c84-bc38-67503fdff504 |
| 8 | Claude Cowork Plugins Explained (& how to build AI employees) | https://www.youtube.com/watch?v=1GZYNyuL6Lg | 3f660432-3efd-4039-ba55-f6b0915879dc |
| 9 | The NEW Nano Banana 2 + Claude Code = $10k Websites | https://www.youtube.com/watch?v=q0TgUtj6vIs | dd1384df-0ad3-4122-aa7a-6a41c1eced4c |
| 10 | This simple Claude Cowork system saves 5 hours a week | https://www.youtube.com/watch?v=W9AOkjmt-XU | af729849-37e1-4828-bc83-b767b9a7066d |
| 11 | I Built An AI Marketing Team With Claude Code (full tutorial) | https://www.youtube.com/watch?v=1JOkxV10Mgk | d5a8a8e2-53ef-4ce2-a8d2-3ee72dcfa34e |
| 12 | 10 Claude Cowork tips I wish I knew from the start | https://www.youtube.com/watch?v=mKBZGcr4pZA | 8b2212da-4c0b-4f55-8716-f39907ee4d5b |
| 13 | Claude Built a Wall Street–Level Financial Model in Excel | https://www.youtube.com/watch?v=DlhRdwZ5XEc | f5d03f04-7085-46f5-abe4-3578c2c6683a |
| 14 | Build BEAUTIFUL Diagrams with Claude Code (Full Workflow) | https://www.youtube.com/watch?v=m3fqyXZ4k4I | 6c894871-50d1-4a15-b754-1d1120f26816 |
| 15 | Stop Paying Anthropic $200/month for Claude Code (Do This Instead) | https://www.youtube.com/watch?v=jJ9jPzPdyDg | b3da177a-4f4f-4d27-906a-2319eaf61ee2 |
| 16 | Claude Code Built a $3000 Voice Agent That Answers Customer Calls | https://www.youtube.com/watch?v=vk2PV5H73x4 | ec1ab1d9-a1a4-4a34-ad22-251933a859ad |
| 17 | I connected Claude Code to Obsidian and it made me 10x more productive | https://www.youtube.com/watch?v=BLdO-32I6Yc | b884224a-3e66-423e-87c9-2dca8c7754c7 |
| 18 | STOP Building AI Agents. Do THIS Instead. | https://www.youtube.com/watch?v=wqH1hTkA6qg | 88345ae4-c9b9-4388-b238-c08faf1d06c4 |
| 19 | Claude Code just became UNSTOPPABLE (Skills 2.0) | https://www.youtube.com/watch?v=Wxf9oqxODU0 | 7e51ad13-de5c-4036-aef8-fbb0d4cff655 |
| 20 | The NEW Nano Banana 2 + Claude Code is UNSTOPPABLE! | https://www.youtube.com/watch?v=cMI2t8M8Pgc | f75725a5-010a-4dbd-8376-d1069bb60a33 |

---

## What Was Implemented

### Session 1 (2026-03-06) — Core Setup

| # | What | File/Location | What It Does | Avg Tokens | Status |
|---|------|---------------|--------------|------------|--------|
| 1 | Global CLAUDE.md | `~/.claude/CLAUDE.md` | Permanent rules loaded into every Claude session on this machine. Covers context discipline, planning, security, parallelism. Claude reads this on every startup — it's your standing instructions. | ~800 input | ✅ |
| 2 | Rules subfolder | `~/.claude/rules/` | Three focused rule files that expand the global CLAUDE.md without bloating it. `context.md` manages memory, `planning.md` enforces plan-before-code, `skills.md` defines the skill standard. | ~300 input each | ✅ |
| 3 | Agents folder | `~/.claude/agents/` | Home for Haiku subagent definitions. Each file tells Claude when to delegate a task to a cheaper/faster model instead of doing it itself. | 0 (just a folder) | ✅ |
| 4 | Stop + Notification hooks | `~/.claude/settings.json` | Shell commands that run automatically when Claude finishes or needs input. Plays a bell + desktop notification so you don't have to watch the terminal. | 0 (OS-level) | ✅ |
| 5 | Verbose mode | `~/.claude/settings.json` | Shows Claude's internal reasoning trace and live token count in the terminal. Helps you understand how Claude thinks and catch token waste early. | +20-50% token visibility | ✅ |
| 6 | Plans folder | `/root/AI/plans/` | Archive of approved plans before execution. Pattern: plan → save → `/clear` → execute. If something breaks, feed the plan back instead of git reverting. | ~500 input (plan doc) | ✅ |
| 7 | Failure tracker | `/root/AI/failures.md` | A table logging every task Claude failed at — date, model, what broke, and when to re-test. Re-test everything after each new model release. | 0 (manual log) | ✅ |

### Session 2 (2026-03-06) — Hardening + Skills

| # | What | File/Location | What It Does | Avg Tokens | Status |
|---|------|---------------|--------------|------------|--------|
| 8 | Permissions (allow/deny) | `~/.claude/settings.json` | Controls which tools Claude can run without asking. Read/Glob/Grep/git/npm/pip auto-approved. Edit/Write still prompt you. `rm -rf` and piped curl installs hard-blocked. | 0 (config) | ✅ |
| 9 | HTTP Proxy sandbox | — | Would intercept all Claude web requests and only allow trusted domains. Prevents prompt injection from malicious websites. | 0 (config) | ⏭️ Skipped |
| 10 | Exa MCP server | `~/.claude/settings.json` + `.env` | Search engine built for AI. Finds recent docs, papers, GitHub issues that Claude's training doesn't have (post-Aug 2025). Free tier: 1,000 searches/month. | ~500 per search | ✅ |
| 11 | VoiceMode MCP | `~/.claude/settings.json` + `.venv` | Lets you speak to Claude instead of typing. Uses OpenAI Whisper for transcription (~$0.006/min). Works in WSL via MCP bridge. Say `converse` to start. | ~200 + Whisper cost | ✅ |
| 12 | Haiku subagents | `~/.claude/agents/` | Three specialist agents running on claude-haiku-4-5 (10x cheaper than Sonnet). `log-reader` for error analysis, `doc-researcher` for lookups, `file-summarizer` for large file overviews. Keeps noise out of your main session. | ~500-1000 (Haiku) vs ~3000+ (Sonnet) | ✅ |
| 13 | `.claude.local.md` | `/root/AI/.claude.local.md` | Machine-specific overrides that never get committed to git. Stores WSL paths, venv location, Windows quirks. Each machine can have its own version. | ~200 input | ✅ |
| 14 | Subfolder CLAUDE.md (tools) | `/root/AI/tools/CLAUDE.md` | Rules that only load when Claude works inside `tools/`. Enforces WAT standards: sys.exit codes, manifest writes, validate_input usage, no tool without a plan. | ~300 input | ✅ |
| 15 | Subfolder CLAUDE.md (workflows) | `/root/AI/workflows/CLAUDE.md` | Rules that only load when Claude works inside `workflows/`. Enforces: never overwrite without asking, naming conventions, 300-line limit. | ~300 input | ✅ |
| 16 | Spec Developer skill | `~/.claude/skills/spec-developer/` | Forces Claude to ask 5-10 clarifying questions before touching any code. Prevents building the wrong thing. Use `/spec-developer` before any new feature. | ~800 input | ✅ |
| 17 | Git Worktree skill | `~/.claude/skills/git-worktree/` | Creates an isolated working directory on a new branch so two Claude sessions can work on the same repo simultaneously without overwriting each other. Use `/git-worktree`. | ~600 input | ✅ |
| 18 | Parallel Sessions skill | `~/.claude/skills/parallel-sessions/` | Guide for running 10-12 Claude sessions at once across different tasks. While one session is thinking, you direct another. Eliminates waiting on AI. Use `/parallel-sessions`. | ~700 input | ✅ |
| 19 | /rewind documented | `~/.claude/rules/planning.md` | Built-in Claude Code command. Rolls back the session to an earlier point if Claude goes off track — faster than manually undoing file changes. | 0 (built-in) | ✅ |
| 20 | Code Bias Fix skill | `~/.claude/skills/code-bias-fix/` | Breaks the cycle of Claude copying bad patterns from existing code. Build the new feature in an empty `/tmp/` folder first (no bias), then integrate back. Use `/code-bias-fix`. | ~600 input | ✅ |

---

## Key Techniques Reference

| Technique | How to Use | Token Cost | Documented In |
|-----------|------------|------------|---------------|
| Plan Mode | Shift+Tab before every feature | ~500-1000 (plan generation) | `~/.claude/rules/planning.md` |
| Spec Developer | `/spec-developer` before coding | ~800 (questions + answers) | `~/.claude/skills/spec-developer/` |
| Context Check | `/context` — see what's eating the window | 0 (built-in) | `~/.claude/rules/context.md` |
| Compact | `/compact` manually at ~40% context | Reduces by ~60% | `~/.claude/rules/context.md` |
| Cost Check | `/cost` — monitor token spend | 0 (built-in) | `~/.claude/rules/context.md` |
| Rewind | `/rewind` — roll back session if off track | 0 (built-in) | `~/.claude/rules/planning.md` |
| Git Worktrees | `/git-worktree` for parallel sessions on same repo | ~600 (skill load) | `~/.claude/skills/git-worktree/` |
| Parallel Sessions | `/parallel-sessions` — 10-12 tabs workflow | ~700 (skill load) | `~/.claude/skills/parallel-sessions/` |
| Haiku Subagents | log-reader, doc-researcher, file-summarizer | ~500-1000 vs 3000+ Sonnet | `~/.claude/agents/` |
| Code Bias Fix | `/code-bias-fix` — build in empty folder first | ~600 (skill load) | `~/.claude/skills/code-bias-fix/` |
| Exa MCP | Ask Claude to search recent docs | ~500 per search | `~/.claude/settings.json` |
| Voice Input | Say `converse` to start voice session | ~200 + $0.006/min Whisper | VoiceMode MCP |
| Skill YAML | Every skill needs name + description front matter | Saves ~900 tokens per skill load | `~/.claude/rules/skills.md` |

---

## Skills to Develop (from AI programming playlist, videos 11–28)

These are not yet implemented — prioritized by impact on current WAT setup.

| Priority | Skill / Topic | Source Video | Implementation Steps (from NotebookLM study) | Status |
|----------|--------------|-------------|----------------------------------------------|--------|
| 🔴 High | **Skills 2.0** — Skill Creator Plugin | #19 | 1. Run `/plugins` in Claude Code terminal → install "Skill Creator Plugin". 2. Type `/skillcreator`, describe the skill. 3. Claude auto-tests it against baseline before saving. 4. Say "save to global skills directory". Key difference: skills are now tested before saving, not just written. | ⏳ Manual (needs Cowork desktop app) |
| 🔴 High | **Ollama free tier** — opt-in local models | #7 | `claude-local` command installed at `/usr/local/bin/claude-local`. Shows model picker. Does NOT change default `claude`. Models available: qwen2.5:7b, llama3.2, mistral:7b. | ✅ Done |
| 🔴 High | **Scheduled Tasks** — unattended WAT pipelines | #6 | Uses Claude Cowork desktop app `/schedule` command. Set prompt + model (Haiku) + frequency. Runs locally. Use for: daily Upwork scan, AI news digest. | ⏳ Ready to configure |
| 🟡 Medium | **Excalidraw Diagram skill** — self-validating diagrams | #14 | Installed from `coleam00/excalidraw-diagram-skill` to `~/.claude/skills/excalidraw-diagram/`. Use: "Create a diagram to explain X". Generates Excalidraw JSON, view at excalidraw.com. | ✅ Done |
| 🟡 Medium | **Obsidian integration** — Claude reads your second brain | #17 | Template at `.tmp/obsidian_CLAUDE.md`. Copy to your Obsidian vault root as `CLAUDE.md`. Then open terminal in vault folder → run `claude`. Edit template with your vault structure. | ✅ Template ready |
| 🟡 Medium | **Claude Cowork framework** — AI employee pattern | #5, #8, #10, #12 | Desktop app (Windows). Download from claude.ai. Has folder access + plugin system + `/schedule`. Complements WAT for desktop workflows. | ⏳ Manual (Windows install) |
| 🟡 Medium | **Voice Agent** — full inbound call system | #16 | Workflow saved to `workflows/voice_agent.md`. Stack: LiveKit + Deepgram + OpenAI + ElevenLabs + Twilio. Run when client found on Upwork. | ✅ Workflow ready |
| 🟢 Low | **Excel automation** — Claude Cowork financial models | #13 | Uses Claude Cowork desktop app. Give it a folder + prompt = `.xlsx` generated automatically. Use when client needs financial model. | ⏳ Manual (needs Cowork app) |
| 🟢 Low | **Stop building agents** — workflow-first thinking | #18 | Already aligned with WAT architecture. | ✅ Aligned |
| 🟢 Low | **Cost reduction** | #15 | Use `claude-local` for low-stakes tasks. Use Haiku subagents for log/doc tasks. | ✅ Done |

---

## How to Add More YouTube Videos
```bash
notebooklm source add "https://www.youtube.com/watch?v=VIDEO_ID" --json
# Wait for processing, then regenerate study guide:
notebooklm generate report --format study-guide --json
```
Then update this file with new learnings and implement anything actionable.
