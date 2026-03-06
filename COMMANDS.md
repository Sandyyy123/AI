# Commands & Tools Reference

> Auto-updated whenever new scripts are added. Last updated: 2026-03-06 (Added TimelinesAI tools: send, fetch, summarize, group_update).

---

## TimelinesAI — WhatsApp Automation

Three tools for automated WhatsApp messaging and conversation summarization via the [TimelinesAI](https://app.timelines.ai) REST API.

**Required .env keys:**
```
TIMELINESAI_API_KEY=<from https://app.timelines.ai/integrations/api/>
TIMELINESAI_SENDER_PHONE=<your connected WhatsApp number, e.g. +491234567890>
```

---

### `timelinesai_send.py` — Send WhatsApp messages (single or bulk)

```bash
# Single message
python tools/timelinesai_send.py --phone +491234567890 --message "Hello!"

# Bulk from CSV (columns: phone,message)
python tools/timelinesai_send.py --csv .tmp/recipients.csv

# Preview without sending
python tools/timelinesai_send.py --phone +491234567890 --message "Hello!" --dry-run
```

**Output:** `.tmp/timelinesai_send_results.json`
**Rate limit:** 2–4s delay between sends (anti-spam)

---

### `timelinesai_fetch_chats.py` — Fetch chat history

```bash
# Fetch all chats (metadata only)
python tools/timelinesai_fetch_chats.py

# Fetch chats + full message history
python tools/timelinesai_fetch_chats.py --with-messages

# Limit to 20 chats with messages
python tools/timelinesai_fetch_chats.py --limit 20 --with-messages

# Only unread chats
python tools/timelinesai_fetch_chats.py --unread-only --with-messages
```

**Output:** `.tmp/timelinesai_chats.json`

---

### `timelinesai_summarize.py` — Summarize conversations with GPT-4o-mini

```bash
# Daily digest of all chats (fetches automatically if no local file)
python tools/timelinesai_summarize.py --mode daily

# Daily digest, fetch fresh data
python tools/timelinesai_summarize.py --mode daily --fetch

# Only summarize unread chats
python tools/timelinesai_summarize.py --mode daily --unread-only --fetch

# Deep summary of a single chat
python tools/timelinesai_summarize.py --mode chat --chat-id <chat_id>

# Use existing local chats file
python tools/timelinesai_summarize.py --mode daily --input .tmp/timelinesai_chats.json
```

**Output:** `.tmp/timelinesai_summary.json` + printed to stdout
**Requires:** `OPENAI_API_KEY` + `TIMELINESAI_API_KEY`

---

### `timelinesai_group_update.py` — Summarize a group and send the update back

One-command pipeline: finds group by name → fetches messages → summarizes → sends update to the group.

```bash
# Full pipeline: summarize + send
python tools/timelinesai_group_update.py --group "6 Months Outkill Engineering"

# Summarize only, no send
python tools/timelinesai_group_update.py --group "6 Months Outkill Engineering" --summarize-only

# Preview without sending
python tools/timelinesai_group_update.py --group "6 Months Outkill Engineering" --dry-run

# Summarize last 50 messages only
python tools/timelinesai_group_update.py --group "6 Months Outkill Engineering" --last 50
```

**Output:** `.tmp/timelinesai_group_update.json` + summary printed to stdout
**Requires:** `TIMELINESAI_API_KEY`, `TIMELINESAI_SENDER_PHONE`, `OPENAI_API_KEY`

---

## How to use this file

All scripts live in `tools/`. Run them from the project root with the `.venv` active:

```bash
cd /root/AI
source .venv/bin/activate
python tools/<script_name>.py --help
```

---

## Zoom → Fireflies Pipeline

These scripts automate the full workflow of finding a Zoom meeting, registering, and sending a Fireflies bot to record it.

---

### `zoom_to_fireflies.py` — Full automated pipeline (with registration)

**What it does:**
The all-in-one command for meetings that require Zoom registration. Chains four steps automatically:
1. Looks up the meeting on Google Calendar by date and organizer
2. Opens a Playwright browser to fill and submit the Zoom registration form
3. Waits for the confirmation email to arrive in Gmail and extracts your personal join link
4. Sends the Fireflies bot ("Fred") to join and record

For meetings longer than 170 minutes, automatically schedules a second bot (Bot 2) to join at T+2h50min, so recordings over 3 hours are fully captured.

**When to use:** Meetings where you need to register first (the Zoom link in the calendar is a `/meeting/register/` URL, not a direct join link).

```bash
python tools/zoom_to_fireflies.py --date 2026-03-08 --organizer "Outskill"
python tools/zoom_to_fireflies.py --date 2026-03-08 --organizer "GEF C4" --time 14:00
python tools/zoom_to_fireflies.py --date 2026-03-08 --organizer "GEF C4" --total-duration 200
```

| Argument | Required | Description |
|----------|----------|-------------|
| `--date` | Yes | Meeting date `YYYY-MM-DD` |
| `--organizer` | Yes | Company or person name to filter calendar events (e.g. `"GEF C4"`) |
| `--time` | No | Meeting time `HH:MM` — narrows results if multiple events that day |
| `--total-duration` | No | Duration in minutes (default: 180). Over 170 triggers auto Bot 2 |

**Requires:** `REGISTRANT_NAME` and `REGISTRANT_EMAIL` in `.env`, Google OAuth set up (`credentials.json`), Playwright, Fireflies API key.

---

### `zoom_to_fireflies_direct.py` — Full automated pipeline (no registration)

**What it does:**
Same as above but for meetings where you're directly invited — no registration required. Skips the registration step entirely and instead searches Gmail for the invite email to get the direct join link.

Steps:
1. Looks up the meeting on Google Calendar
2. Searches Gmail for the invite email and extracts the direct join link (`zoom.us/j/...`)
3. Sends the Fireflies bot to join and record

**When to use:** Internal meetings, recurring meetings, or any meeting where the Zoom link is a direct join URL (`zoom.us/j/...`) rather than a registration page.

```bash
python tools/zoom_to_fireflies_direct.py --date 2026-03-08 --organizer "GEF C4"
python tools/zoom_to_fireflies_direct.py --date 2026-03-08 --organizer "GEF C4" --time 14:00
```

| Argument | Required | Description |
|----------|----------|-------------|
| `--date` | Yes | Meeting date `YYYY-MM-DD` |
| `--organizer` | Yes | Organizer or company name to filter results |
| `--time` | No | Meeting time `HH:MM` — narrows calendar results |
| `--total-duration` | No | Duration in minutes (default: 180) |

---

### `get_zoom_from_calendar.py` — Extract Zoom link from Google Calendar

**What it does:**
Queries Google Calendar for a specific date (and optionally time/organizer) and extracts the Zoom meeting URL, title, and description. Searches the event's `hangoutLink`, `location`, and `description` fields for any Zoom URL. If the matched event has no description, it scans other events on the same day for a related event that shares the same Zoom URL and pulls the description from there.

Used as a building block by the pipeline scripts, but can also be run standalone to look up meeting details.

```bash
python tools/get_zoom_from_calendar.py --date 2026-03-08
python tools/get_zoom_from_calendar.py --date 2026-03-08 --time 14:00 --organizer "Outskill"
```

| Argument | Required | Description |
|----------|----------|-------------|
| `--date` | Yes | Date to search `YYYY-MM-DD` |
| `--time` | No | Time `HH:MM` — searches ±15 min around this time |
| `--organizer` | No | Filter by organizer name or email |

**Output:** Prints meeting title, organizer, time, Zoom URL, and description. Also prints `ZOOM_URL=`, `MEETING_TITLE=`, `MEETING_DESC=` lines for easy parsing.

---

### `register_zoom_meeting.py` — Register for a Zoom meeting

**What it does:**
Uses Playwright (headless Chromium) to navigate to a Zoom registration page, fill in the first name, last name, and email fields, and click submit. Takes screenshots at each step (saved to `.tmp/`) so you can debug if registration fails. Zoom sends a confirmation email with your personal join link after successful registration.

```bash
python tools/register_zoom_meeting.py --url "https://zoom.us/meeting/register/..."
python tools/register_zoom_meeting.py --url "https://zoom.us/meeting/register/..." --name "John Doe" --email "john@example.com"
```

| Argument | Required | Description |
|----------|----------|-------------|
| `--url` | Yes | Zoom registration page URL |
| `--name` | No | Full name (defaults to `REGISTRANT_NAME` in `.env`) |
| `--email` | No | Email address (defaults to `REGISTRANT_EMAIL` in `.env`) |

---

### `get_zoom_link_from_gmail.py` — Extract personal join link from Gmail (post-registration)

**What it does:**
After registering for a Zoom meeting, searches Gmail for the confirmation email sent by `no-reply@zoom.us` and extracts the personal join link (`zoom.us/j/...`). Waits a configurable number of seconds before searching (default 10s) to give the email time to arrive, and retries up to 3 times with 15-second pauses if the email isn't found immediately.

```bash
python tools/get_zoom_link_from_gmail.py --title "GEF C4 Session"
python tools/get_zoom_link_from_gmail.py --title "GEF C4 Session" --wait 30
```

| Argument | Required | Description |
|----------|----------|-------------|
| `--title` | Yes | Meeting title to search for in email subject |
| `--wait` | No | Seconds to wait before searching (default: 10) |

---

### `get_direct_zoom_from_gmail.py` — Extract direct Zoom link from a Gmail invite

**What it does:**
Searches Gmail for an email containing a direct Zoom join link (`zoom.us/j/` or `zoom.us/w/`). Use this when you were directly invited to a meeting and received the join link in an email — no registration was required. Tries multiple search queries from most specific to broadest: subject match, sender match, date range, then a 30-day fallback.

```bash
python tools/get_direct_zoom_from_gmail.py --title "Team Standup"
python tools/get_direct_zoom_from_gmail.py --organizer "Outskill" --date 2026-03-08
python tools/get_direct_zoom_from_gmail.py --title "Team Standup" --organizer "Outskill" --date 2026-03-08
```

| Argument | Required | Description |
|----------|----------|-------------|
| `--title` | No* | Meeting title to search in email subject |
| `--organizer` | No* | Sender/organizer name or domain to search |
| `--date` | No* | Date `YYYY-MM-DD` — searches emails ±7 days around this date |

*At least one argument is required.

---

### `add_to_fireflies.py` — Send Fireflies bot to a Zoom meeting

**What it does:**
Calls the Fireflies GraphQL API (`addToLiveMeeting`) to dispatch "Fred" (the Fireflies recording bot) to a Zoom meeting URL. Automatically derives a clean recording title from the calendar event description by stripping Zoom boilerplate lines (join links, Meeting ID, passcode, dial-in numbers).

For meetings longer than 170 minutes, spawns a background process (survives terminal closure) that sleeps until T+2h50min, then sends a second bot for the final 30 minutes — ensuring full coverage of 3-hour sessions.

```bash
python tools/add_to_fireflies.py --url "https://zoom.us/j/..." --title "GEF C4" --description "Topic: RAG pipelines"
python tools/add_to_fireflies.py --url "https://zoom.us/j/..." --title "Long Meeting" --total-duration 200
```

| Argument | Required | Description |
|----------|----------|-------------|
| `--url` | Yes | Direct Zoom join URL (`zoom.us/j/...`) |
| `--title` | No | Fallback recording name if description is empty |
| `--description` | No | Calendar event description — used as primary recording name |
| `--total-duration` | No | Meeting duration in minutes (default: 180). >170 triggers Bot 2 |
| `--no-schedule` | No | Disable Bot 2 scheduling (used internally by Bot 2 itself) |

**Rate limit:** Fireflies allows 3 API calls per 20 minutes. Bot 1 and Bot 2 are spaced 170 minutes apart, so no issue in normal use.

---

## Video / Camtasia Tools

---

### `convert_to_mp4.py` — Convert video files to MP4 using ffmpeg

**What it does:**
Batch converts video files (`.mov`, `.mkv`, `.wmv`, `.avi`) to MP4 using ffmpeg. Saves output files alongside the originals with the same filename but `.mp4` extension. Skips files that already have an `.mp4` counterpart. Skips `.trec` files (Camtasia-native format that must be exported from within Camtasia itself — see `conver_camtasia_files.py` or `export_camtasia_trec.py`).

Default codec is `libx264` (H.264) with CRF 23 (good quality/size balance) and `aac` audio.

```bash
python tools/convert_to_mp4.py --dir "/mnt/c/Users/grove/OneDrive/Documentos/Camtasia"
python tools/convert_to_mp4.py --file "/mnt/c/path/to/recording.mov"
python tools/convert_to_mp4.py --dir "/mnt/c/..." --codec libx265
python tools/convert_to_mp4.py --dir "/mnt/c/..." --recursive
```

| Argument | Required | Description |
|----------|----------|-------------|
| `--dir` | Yes* | Directory to scan for video files |
| `--file` | Yes* | Single video file to convert |
| `--codec` | No | Video codec (default: `libx264`). Options: `libx265`, `copy` |
| `--recursive` | No | Also scan subdirectories |

*`--dir` or `--file` required (mutually exclusive).

**Requires:** `ffmpeg` installed (`sudo apt install ffmpeg`).

---

### `conver_camtasia_files.py` — Export Camtasia `.trec` files to MP4 via GUI automation

**What it does:**
Automates the Camtasia GUI on Windows using AutoHotkey v2 to export `.trec` recordings to MP4 without any manual clicking. For each file it:
1. Opens Camtasia with the `.trec` file
2. Waits for the project to fully load (15 second buffer for large files)
3. Triggers `File > Export > Local File...` via the menu
4. Clicks through the export wizard (5× Enter)
5. Waits for rendering to complete (up to 1 hour)
6. Moves the output MP4 to the expected path (alongside the `.trec` file)
7. Closes Camtasia and moves to the next file

Shows a tooltip progress indicator during processing. Runs from WSL and controls the Windows GUI.

```bash
python tools/conver_camtasia_files.py --dir "/mnt/c/Users/grove/OneDrive/Documentos/Camtasia"
python tools/conver_camtasia_files.py --file "/mnt/c/Users/grove/OneDrive/Documentos/Camtasia/Rec 02-21-26.trec"
```

| Argument | Required | Description |
|----------|----------|-------------|
| `--dir` | Yes* | Directory containing `.trec` files |
| `--file` | Yes* | Single `.trec` file to export |

**Requires:** AutoHotkey v2 (`winget install AutoHotkey.AutoHotkey`), Camtasia installed. Do NOT touch keyboard/mouse while it runs.

---

### `export_camtasia_trec.py` — Export Camtasia `.trec` files to MP4 (alternative version)

**What it does:**
Functionally identical to `conver_camtasia_files.py` — uses AutoHotkey v2 to automate Camtasia GUI export of `.trec` files to MP4. This is an alternative/updated version of the same script. Supports both single-file and directory batch processing.

```bash
python tools/export_camtasia_trec.py --dir "/mnt/c/Users/grove/OneDrive/Documentos/Camtasia"
python tools/export_camtasia_trec.py --file "/mnt/c/Users/grove/OneDrive/Documentos/Camtasia/Rec 02-21-26.trec"
```

| Argument | Required | Description |
|----------|----------|-------------|
| `--dir` | Yes* | Directory containing `.trec` files |
| `--file` | Yes* | Single `.trec` file to export |

---

## PDF Tools

---

### `save_course_as_pdf.py` — Save course lessons from dailydoseofds.com as PDFs

**What it does:**
Uses Playwright to navigate dailydoseofds.com, log in (you handle login in the browser window it opens), discover all course links on the homepage, then visit each lesson page and save it as a PDF. For each lesson it:
1. Navigates to the lesson URL
2. Scrolls through the page to trigger lazy image loading
3. Waits for all images to finish loading
4. Saves the page as a PDF named `01_Lesson_Title.pdf` inside a folder named after the course

Skips lessons that already have a PDF (safe to re-run after interruption). Detects and skips paywalled/locked lessons.

Output goes to `C:/Users/grove/Downloads/courses/` (Windows path).

```bash
python tools/save_course_as_pdf.py                          # all courses
python tools/save_course_as_pdf.py --course "MCP"           # one course by keyword
python tools/save_course_as_pdf.py --course "RAG"
python tools/save_course_as_pdf.py --debug-links            # print all links found (for debugging)
```

| Argument | Required | Description |
|----------|----------|-------------|
| `--url` | No | Start URL (default: `https://www.dailydoseofds.com/`) |
| `--output` | No | Output directory (default: `C:/Users/grove/Downloads/courses`) |
| `--course` | No | Filter to one course by keyword in title or URL |
| `--debug-links` | No | Print all links found on each course page (helps fix sidebar detection) |

---

### `pdf_merge_enhance.py` — Merge PDFs with watermark and similarity reordering

**What it does:**
Takes a folder of PDFs and merges them into a single file with three enhancements applied to every page:
1. **Footer removal** — draws a white rectangle over the bottom 36pt strip to cover NotebookLM footers
2. **Diagonal watermark** — adds a subtle "Generated by Sandeep Grover" watermark at 35° rotation
3. **Similarity reordering** — uses sentence embeddings (`all-MiniLM-L6-v2`) and nearest-neighbour traversal to reorder pages so related content appears together (groups similar slides across different PDFs)

```bash
python tools/pdf_merge_enhance.py --input ./courses/RAG_Crash_Course
python tools/pdf_merge_enhance.py --input ./courses/RAG_Crash_Course --output ~/Desktop/rag_merged.pdf
python tools/pdf_merge_enhance.py --input ./courses/RAG_Crash_Course --no-reorder
```

| Argument | Required | Description |
|----------|----------|-------------|
| `--input` | Yes | Folder containing PDFs to merge |
| `--output` | No | Output PDF path (default: `Downloads/<folder_name>_merged.pdf`) |
| `--no-reorder` | No | Skip similarity reordering, keep original alphabetical order |

**Requires:** `pdfplumber`, `pypdf`, `reportlab`, `sentence-transformers`, `scikit-learn`.

---

### `pdf_editor_app.py` — Standalone GUI PDF Editor (packages to .exe)

**What it does:**
A desktop PDF editor built with tkinter + PyMuPDF. Runs locally — no browser, no internet needed.
Ships as a single `.exe` so anyone can install it on Windows without Python.

**Features:**
- Open & view PDFs with page thumbnails sidebar
- Navigate pages (prev/next, click thumbnail, type page number)
- Delete pages, rotate pages (CW / CCW 90°)
- Merge another PDF into the current document
- Extract a page range to a new PDF
- Add text watermark to all pages (diagonal, customizable text)
- Zoom in/out/fit, keyboard shortcuts

**Run directly (development):**
```bash
pip install pymupdf pillow
python tools/pdf_editor_app.py
```

**Build .exe on Windows:**
1. Copy `tools/pdf_editor_app.py` and `tools/build_pdf_editor.bat` to a Windows folder
2. Double-click `build_pdf_editor.bat`
3. Find `dist/PDF Editor.exe` — share this single file with anyone

```
tools/pdf_editor_app.py      # The app
tools/build_pdf_editor.bat   # Windows build script (uses PyInstaller)
```

**Requires (auto-installed by build script):** `pymupdf`, `pillow`, `pyinstaller`

---

### `html_to_pdf_app.py` — All-in-one PDF toolkit (ilovepdf-style web app)

**What it does:**
A Flask web application with 18 PDF tools organized into categories, accessible at http://localhost:5050. Features a dark card-based UI with category filters. Click any tool card to reveal a drop zone + options panel. Converts and auto-downloads the result.

**Tools included:**

| Category | Tools |
|----------|-------|
| Organize | Merge PDF, Split PDF, Remove Pages, Extract Pages |
| Optimize | Compress PDF |
| Convert to PDF | HTML to PDF, Image to PDF, Word to PDF, Excel to PDF, PowerPoint to PDF |
| Convert from PDF | PDF to JPG, PDF to Word, PDF to Excel |
| Edit | Rotate PDF, Add Page Numbers, Add Watermark |
| Security | Protect PDF (add password), Unlock PDF (remove password) |

```bash
python tools/html_to_pdf_app.py
# Then open http://localhost:5050 in your browser
```

No arguments — runs on port 5050. Max file size: 50 MB.

**Requires:** `flask`, `weasyprint`, `pypdf`, `Pillow`, `pdf2image`, `reportlab`, `poppler-utils` (system).
Optional for PDF↔Office: `libreoffice` (system), `pdf2docx`, `tabula-py`.

**Install all:**
```bash
pip install flask weasyprint pypdf Pillow pdf2image reportlab
sudo apt install poppler-utils libreoffice
```

---

## NotebookLM Tools

---

### `create_notebooklm_notebooks.py` — Bulk-create NotebookLM notebooks from course folders

**What it does:**
Iterates over every subfolder in `./courses/`, creates a NotebookLM notebook named after the folder, and uploads all `.pdf` and `.txt` files in that folder as sources. Retries failed uploads once with a 3-second delay. Prints a summary table at the end showing how many files were uploaded vs failed per notebook.

```bash
python tools/create_notebooklm_notebooks.py
python tools/create_notebooklm_notebooks.py --courses-dir /root/AI/courses
python tools/create_notebooklm_notebooks.py --folders RAG_Crash_Course Multiomics_T1
```

| Argument | Required | Description |
|----------|----------|-------------|
| `--courses-dir` | No | Path to courses directory (default: `./courses`) |
| `--folders` | No | Only process specific folder names (space-separated) |

**Requires:** `notebooklm` CLI authenticated (`notebooklm login`).

---

### `generate_notebooklm_artifacts.py` — Generate artifacts from NotebookLM notebooks

**What it does:**
Interactive tool that walks you through generating multiple artifact types from a NotebookLM notebook. You pick:
- Which notebook to use (shows numbered list)
- Which sources within the notebook to use (or all)
- Artifact types: `audio` (podcast), `slide-deck`, `infographic`
- Prompt style: `conceptual` (theory/intuition/mental models) or `script` (implementation/code/practical steps) or both
- How many variations to generate per artifact/style combination
- How many to run in parallel

Uses pre-written prompt templates tuned for each artifact/style combination. Downloads completed artifacts to `C:/Users/grove/Downloads/notebooklm_artifacts/<notebook_title>/`. Filenames include type, style, and run number (e.g. `01_podcast_conceptual_RAG_Crash_Course.mp3`).

```bash
python tools/generate_notebooklm_artifacts.py
python tools/generate_notebooklm_artifacts.py --notebook "RAG"   # partial ID or title match
```

| Argument | Required | Description |
|----------|----------|-------------|
| `--notebook` | No | Notebook ID or partial title match. If omitted, shows interactive list |

**Requires:** `notebooklm` CLI authenticated. Generation can take 10–45 minutes per artifact.

---

## YouTube AI Agent Learning Tracker

Pipeline to find, transcribe, and learn from the latest YouTube videos on AI agents. Run `/youtube-ai-tracker` or chain the four tools manually.

---

### `fetch_youtube_ai_videos.py` — Fetch latest AI agent videos via YouTube RSS

Searches YouTube across 7 default AI/agent keyword queries. No API key needed. Deduplicates across runs.

```bash
python tools/fetch_youtube_ai_videos.py                           # all default queries
python tools/fetch_youtube_ai_videos.py --query "CrewAI agents"  # single keyword
python tools/fetch_youtube_ai_videos.py --days 7                  # last 7 days
python tools/fetch_youtube_ai_videos.py --dry-run
```
Output: `.tmp/youtube_videos.json`

---

### `extract_youtube_transcript.py` — Download video transcripts (free)

Fetches transcripts via `youtube-transcript-api`. Tries manual English → auto-generated English → any language. Skips gracefully if unavailable.

```bash
python tools/extract_youtube_transcript.py
python tools/extract_youtube_transcript.py --dry-run
python tools/extract_youtube_transcript.py --max-words 8000
```
Output: `.tmp/youtube_transcripts.json`

---

### `summarize_video_insights.py` — Extract insights using Claude Haiku

Per video: summary, key lessons, tools mentioned, techniques, difficulty, monetization tips, action item, `worth_watching` flag, **`relevance_score` (0–10)**, `claude_code_relevant` flag, and `key_timestamps`. Updates cumulative `tools_mentioned.json` tracker.

Scoring guide: 9-10 = specific tools + real code + novel techniques; 7-8 = named tools + actionable; 5-6 = general but useful; <5 = vague/filler.

```bash
python tools/summarize_video_insights.py
python tools/summarize_video_insights.py --top 5    # top 5 by transcript length
python tools/summarize_video_insights.py --dry-run
```
Output: `.tmp/youtube_insights.json` + `.tmp/tools_mentioned.json`
Cost: ~$0.002/video (Claude Haiku)

---

### `extract_learning_notes.py` — Build Claude Code study notes from top videos

Filters videos with `relevance_score >= 7` (or `claude_code_relevant=true`), then uses Claude Haiku to extract: core concept, Claude Code skills, agent patterns, commands/syntax, timestamped key moments, mistakes to avoid, and a build idea. Outputs a structured Markdown study guide.

```bash
python tools/extract_learning_notes.py                  # relevance >= 7
python tools/extract_learning_notes.py --min-score 6    # lower threshold
python tools/extract_learning_notes.py --all            # all videos regardless of score
python tools/extract_learning_notes.py --dry-run        # print first 3000 chars
```

Output: `.tmp/claude_code_notes_YYYY-MM-DD.md` + `.tmp/learning_moments.json`
Cost: ~$0.003/video (Claude Haiku)

**Run after `summarize_video_insights.py`.**

---

### `save_learning_digest.py` — Compile daily markdown digest

Produces a dated digest with: tool trends, recommended videos with full breakdown, consolidated action items, and low-value video list.

```bash
python tools/save_learning_digest.py
python tools/save_learning_digest.py --worth-only   # only recommended videos
python tools/save_learning_digest.py --dry-run      # print to terminal only
```
Output: `.tmp/ai_learning_digest_YYYY-MM-DD.md`

---

### Full pipeline

```bash
cd /root/AI && source .venv/bin/activate && \
python tools/fetch_youtube_ai_videos.py && \
python tools/extract_youtube_transcript.py && \
python tools/summarize_video_insights.py && \
python tools/extract_learning_notes.py && \
python tools/save_learning_digest.py
```

**Install deps (first time):**
```bash
pip install feedparser requests youtube-transcript-api anthropic python-dotenv
```

---

## Upwork Automated Job Applier

Pipeline to find, score, and apply to Upwork jobs. Run `/upwork-applier` or chain the four tools manually.

---

### `fetch_upwork_jobs.py` — Fetch new Upwork listings via RSS

Fetches jobs across 7 default AI/tech keyword queries. Deduplicates against a seen-jobs cache so only new listings are returned each run.

```bash
python tools/fetch_upwork_jobs.py                          # all default queries
python tools/fetch_upwork_jobs.py --query "RAG pipeline"   # single keyword
python tools/fetch_upwork_jobs.py --min-budget 500         # filter by budget
python tools/fetch_upwork_jobs.py --dry-run                # print without saving
python tools/fetch_upwork_jobs.py --reset-cache            # re-fetch all (ignore seen)
```

Output: `.tmp/upwork_jobs.json`

---

### `score_upwork_job.py` — Score jobs for fit using Claude Haiku

Reads `.tmp/upwork_jobs.json`, scores each job 0–10 against the candidate profile, adds red flags and recommended bid price.

```bash
python tools/score_upwork_job.py                 # default threshold 7
python tools/score_upwork_job.py --threshold 6   # lower the bar
python tools/score_upwork_job.py --dry-run        # print scores only
```

Output: `.tmp/upwork_scored.json` (sorted by score)
Cost: ~$0.001/job (Claude Haiku)

---

### `generate_upwork_proposal.py` — Generate tailored proposals using Claude Sonnet

Generates a proposal for each job scoring >= threshold. Proposals are under 130 words, open with the client's problem, and end with a fixed CTA.

```bash
python tools/generate_upwork_proposal.py                   # all 7+ jobs
python tools/generate_upwork_proposal.py --top 5           # top 5 only
python tools/generate_upwork_proposal.py --threshold 6     # lower threshold
python tools/generate_upwork_proposal.py --dry-run         # print without saving
```

Output: `.tmp/upwork_proposals.json` + terminal printout
Cost: ~$0.01/proposal (Claude Sonnet)

---

### `track_upwork_applications.py` — Log proposals to Google Sheets

Upserts proposals into "Upwork Applications" Google Sheet. Creates the sheet if it doesn't exist. Deduplicates by URL.

```bash
python tools/track_upwork_applications.py                           # write to Sheets
python tools/track_upwork_applications.py --dry-run                 # print rows only
python tools/track_upwork_applications.py --status "Applied"        # set custom status
```

Output: Google Sheets "Upwork Applications"
Requires: `credentials.json` + `token.json` (Google OAuth)

---

### Full pipeline (one command)

```bash
cd /root/AI && source .venv/bin/activate && \
python tools/fetch_upwork_jobs.py && \
python tools/score_upwork_job.py && \
python tools/generate_upwork_proposal.py && \
python tools/track_upwork_applications.py
```

**Required `.env` variables:**
```
UPWORK_PROFILE_SUMMARY="..."
UPWORK_HOURLY_RATE=45
UPWORK_PORTFOLIO="..."
```

---

## Jay Shah Meeting Tools

Tools to summarize email history and create a Calendly invite. Run them in sequence.

---

### `summarize_jay_shah_emails.py` — Summarize last 30 days of emails from Jay Shah

Fetches all emails from Jay Shah via Gmail API, sends them to GPT-4o, and produces a structured Markdown summary with: who Jay Shah is, key topics, action items, open questions, a suggested meeting agenda, and recommended meeting duration.

```bash
python tools/summarize_jay_shah_emails.py
```

Output: terminal + `.tmp/jay_shah_email_summary.md`

**Requires:** Google OAuth (`credentials.json` + `token.json`), `OPENAI_API_KEY` in `.env`

---

### `create_calendly_invite.py` — Create a one-time Calendly scheduling link

Fetches your Calendly event types, picks the best duration (auto-detected from the email summary or specified manually), and creates a single-use booking link to share with Jay Shah.

```bash
python tools/create_calendly_invite.py                  # auto-pick duration from email summary
python tools/create_calendly_invite.py --duration 30    # force 30-min event type
python tools/create_calendly_invite.py --list-types     # just list event types
```

| Argument | Required | Description |
|----------|----------|-------------|
| `--duration` | No | Preferred event duration in minutes (30, 45, 60). Falls back to shortest type |
| `--list-types` | No | Print available Calendly event types and exit |

Output: terminal + `.tmp/jay_shah_calendly_link.txt`

**Requires:** `CALENDLY_API_KEY` in `.env`

---

### `create_zoom_meeting.py` — Create a Zoom meeting link

Creates a Zoom meeting via the Zoom API and prints the join URL. Works for any meeting with any attendee.

```bash
python tools/create_zoom_meeting.py                                       # defaults: "Meeting", 60 min
python tools/create_zoom_meeting.py --topic "Jay Shah Kickoff" --duration 60
python tools/create_zoom_meeting.py --topic "Team Sync" --duration 30 --start "2026-03-10T14:00:00" --timezone "America/New_York"
```

| Argument | Required | Description |
|----------|----------|-------------|
| `--topic` | No | Meeting title (default: "Meeting") |
| `--duration` | No | Duration in minutes (default: 60) |
| `--start` | No | Start time in ISO 8601 format, e.g. `2026-03-10T14:00:00` (default: ASAP) |
| `--timezone` | No | Timezone string (default: `UTC`). Example: `America/New_York` |

Output: terminal + `.tmp/zoom_meeting_link.txt`

**Requires in `.env`:**
- `ZOOM_ACCOUNT_ID`
- `ZOOM_CLIENT_ID`
- `ZOOM_CLIENT_SECRET`

Setup: marketplace.zoom.us → Create "Server-to-Server OAuth" app → Add scope `meeting:write` → copy the 3 keys.

---

### Full workflow

```bash
cd /root/AI && source .venv/bin/activate
python tools/summarize_jay_shah_emails.py   # step 1: read emails + generate agenda
python tools/create_calendly_invite.py      # step 2: create invite link
```

---

## Grover Tools Sync

### `sync_grover_tools.py` — Regenerate scriptSources.ts and push to GitHub

Reads every Python script listed in `SCRIPT_ORDER`, escapes it for embedding in a JS template literal, writes `src/lib/scriptSources.ts` in the Lovable project, then commits and pushes to GitHub so Grover Tools (sandyyysfileeditor.lovable.app) stays in sync.

**Run this whenever you add or update a Python tool.**

```bash
python tools/sync_grover_tools.py              # regenerate + push
python tools/sync_grover_tools.py --dry-run    # regenerate file only, no git
```

**To add a new script:**
1. Add the filename to `SCRIPT_ORDER` in `sync_grover_tools.py`
2. Add the import + `code:` property in `ScriptsReference.tsx`
3. Run `python tools/sync_grover_tools.py`

---

## WAT Infrastructure Tools

Shared helpers used by all pipelines. Import as modules or run standalone.

---

### `validate_input.py` — Input sanitization helper

Strips shell metacharacters from string args before they reach API calls or subprocesses.

```python
from tools.validate_input import sanitize, sanitize_url, sanitize_path, sanitize_date, sanitize_time
```

| Function | Use for |
|----------|---------|
| `sanitize(s)` | Titles, organizer names, descriptions — strips `; & | $ < > { }` |
| `sanitize_url(url)` | Zoom URLs, API endpoints — validates https:// and no shell chars |
| `sanitize_path(p)` | File/dir paths — resolves and validates, optional `must_exist=True` |
| `sanitize_date(s)` | YYYY-MM-DD format validation |
| `sanitize_time(s)` | HH:MM format validation |

---

### `check_env.py` — .env key existence checker

Checks only for key *presence* (never reads or prints values). Safe to run any time.

```bash
python tools/check_env.py                      # all pipelines
python tools/check_env.py --pipeline upwork
python tools/check_env.py --pipeline zoom
python tools/check_env.py --all
```

---

### `manifest.py` — .tmp/ pipeline manifest

Records which steps completed. Dependent tools call `assert_step_done()` to block out-of-order runs.

```bash
python tools/manifest.py           # show status table
python tools/manifest.py --clear   # wipe manifest (start fresh)
```

```python
from tools.manifest import write_entry, assert_step_done
assert_step_done("fetch_upwork_jobs")   # exits 1 if prerequisite not done
write_entry("my_tool", outputs=["file.json"], count=12)
```

---

### `preflight.py` — Pre-flight environment check

Validates keys, OAuth files, token expiry, Python deps, and .tmp/ writability before any pipeline run.

```bash
python tools/preflight.py                      # check all pipelines
python tools/preflight.py --pipeline upwork
python tools/preflight.py --pipeline zoom
python tools/preflight.py --fix               # auto-refresh expiring OAuth token
```

Exit: `0` = all passed, `1` = failures found.

---

### `run_pipeline.py` — Pipeline orchestrator

Single-command entry point for any pipeline. Runs preflight, then chains tools in order with proper exit-code handling.

```bash
python tools/run_pipeline.py upwork
python tools/run_pipeline.py upwork --dry-run --threshold 6 --top 5
python tools/run_pipeline.py upwork --force --skip-preflight
python tools/run_pipeline.py zoom --date 2026-03-08 --organizer "Outskill"
python tools/run_pipeline.py zoom --date 2026-03-08 --organizer "GEF C4" --direct
python tools/run_pipeline.py status           # show manifest table
python tools/run_pipeline.py manifest --clear
```

| Argument | Pipeline | Description |
|----------|----------|-------------|
| `--dry-run` | upwork | Print only, no file writes |
| `--force` | upwork | Re-run even if output files exist |
| `--threshold N` | upwork | Min score to generate proposals (default: 7) |
| `--top N` | upwork | Only generate proposals for top N jobs |
| `--skip-preflight` | both | Bypass preflight checks |
| `--direct` | zoom | Direct-invite flow (no Zoom registration) |
| `--total-duration N` | zoom | Meeting length in minutes (default: 180) |

---

## Skills (`/skill` commands)

Skills are invoked automatically by Claude based on intent, or explicitly with `/skill-name`.

| Skill | Trigger | Description |
|-------|---------|-------------|
| `/preflight` | "check if ready", "are my keys set?", "validate setup" | Pre-flight: keys, OAuth, token expiry, deps, .tmp/ writability |
| `/pipeline-status` | "what has run?", "show manifest", "what's in .tmp?" | Shows .tmp/manifest.json — steps completed, counts, timestamps |
| `/morning-brief` | "morning check", "any new Upwork jobs?", "quick scan" | Fetch + score only — no proposals, ~$0.02, fast daily check |
| `/youtube-ai-tracker` | "track YouTube AI videos", "what's new in AI agents on YouTube", "learn from AI agent videos" | Fetches latest AI agent YouTube videos, extracts transcripts, summarizes insights, saves daily digest |
| `/upwork-applier` | "find Upwork jobs", "generate Upwork proposals", "check Upwork" | Full pipeline: fetch → score → proposals → track to Google Sheets |
| `/notebooklm` | "create a podcast", "generate a quiz", or any NotebookLM task | Full NotebookLM API — create notebooks, add sources, generate all artifact types |
| `/notebooklm-courses` | "upload courses to NotebookLM", "create notebooks from courses" | Creates one NotebookLM notebook per course folder and uploads all PDFs as sources |
| `/html-to-pdf` | "convert HTML to PDF", "export page as PDF", "save as PDF" | Guides conversion using `weasyprint` (HTML files), `playwright` (URLs/JS-heavy), or `wkhtmltopdf` (system tool) |

---

## RAG Improvement Pipeline

Tools for studying reference GitHub repos, identifying RAG technique gaps, generating scripts, and chatting with Claude using all repos as context. Full SOP: `workflows/rag_improvement.md`.

---

### `rag_dashboard.py` — Terminal dashboard for current RAG state

Shows cached repos, coverage %, gaps, generated scripts, and recent pipeline steps.

```bash
python tools/rag_dashboard.py
```

---

### `fetch_github_repo.py` — Fetch and cache a GitHub repo

Downloads full file tree + content to `.tmp/repos/<owner>__<repo>/`. Supports partial fetches.

```bash
python tools/fetch_github_repo.py --repo Sandyyy123/GenAIEngineering-Cohort3
python tools/fetch_github_repo.py --repo NirDiamant/RAG_Techniques --tree-only   # fast scan first
python tools/fetch_github_repo.py --repo NirDiamant/RAG_Techniques --force        # re-fetch
```

| Argument | Description |
|----------|-------------|
| `--repo` | `owner/repo` format |
| `--tree-only` | Only fetch file tree, not content (faster for large repos) |
| `--force` | Re-fetch even if already cached |

**Requires:** `GITHUB_TOKEN` in `.env` (avoids rate limits).

---

### `browse_repo.py` — Terminal browser for cached repos

List files, search for keywords, or view a specific file from a cached repo.

```bash
python tools/browse_repo.py --list                                               # all cached repos
python tools/browse_repo.py --repo Sandyyy123/GenAIEngineering-Cohort3 --path . # list root
python tools/browse_repo.py --repo Sandyyy123/GenAIEngineering-Cohort3 --search "chunking"
python tools/browse_repo.py --repo Sandyyy123/GenAIEngineering-Cohort3 --file src/chunker.py
```

---

### `analyze_rag_concepts.py` — Scan repo for RAG gaps

Compares 26 tracked RAG concepts against `/root/AI` codebase. Saves gap report to `.tmp/rag_gap_report.json`. Uses GPT-4o-mini for priority recommendations.

```bash
python tools/analyze_rag_concepts.py --repo Sandyyy123/GenAIEngineering-Cohort3
python tools/analyze_rag_concepts.py --repo Sandyyy123/GenAIEngineering-Cohort3 --dry-run
```

Output: `.tmp/rag_gap_report.json`
Cost: ~$0.001 (GPT-4o-mini)

---

### `generate_rag_script.py` — Generate a Python script for a RAG gap

Takes a technique from the gap report, reads reference code, uses GPT-4o to generate a production-ready script.

```bash
python tools/generate_rag_script.py --list                            # see available gaps
python tools/generate_rag_script.py --technique semantic_chunking
python tools/generate_rag_script.py --technique reranking --dry-run
```

Output: `tools/generated/<technique>.py`
Cost: ~$0.02/script (GPT-4o)

---

### `chat_with_repos.py` — Chat with Claude using all cached repos as context

Interactive (or single-question) chat. Searches all 12 cached repos + class notebooks for files relevant to your question, then streams a GPT-4o response.

```bash
python tools/chat_with_repos.py                                      # interactive mode
python tools/chat_with_repos.py --question "What chunking am I missing?"
python tools/chat_with_repos.py --repos RAG_Techniques ragas         # filter repos
```

| Command (interactive) | Action |
|-----------------------|--------|
| `quit` | Exit |
| `clear` | Reset conversation history |
| `repos` | List available cached repos |

**Context limits:** Top 5 files per repo, max 80K chars total. History kept to last 10 messages.

Available repos: `RAG_Techniques`, `RAG-Anything`, `ragas`, `graphrag`, `dspy`, `mem0`, `GenAI_Agents`, `ragflow`, `dify`, `OpenScholar`, `Prompt_Engineering`, `GenAIEngineering`

---

### Full RAG improvement pipeline

```bash
cd /root/AI && source .venv/bin/activate
python tools/rag_dashboard.py
python tools/fetch_github_repo.py --repo Sandyyy123/GenAIEngineering-Cohort3
python tools/analyze_rag_concepts.py --repo Sandyyy123/GenAIEngineering-Cohort3
python tools/generate_rag_script.py --list
python tools/chat_with_repos.py
```

---

## Workflows (`workflows/`)

Markdown SOPs that describe when and how to use the tools above.

| File | Description |
|------|-------------|
| `workflows/zoom_to_fireflies.md` | Full SOP for the Zoom → Fireflies recording pipeline |
| `workflows/camtasia_to_mp4.md` | SOP for exporting Camtasia `.trec` recordings to MP4 |
| `workflows/notebooklm_pipeline.md` | SOP for building a NotebookLM research and artifact pipeline |
| `workflows/upwork_job_applier.md` | SOP for the Upwork automated job applier pipeline |
| `workflows/rag_improvement.md` | SOP for the full RAG improvement pipeline (fetch → analyze → generate → chat) |

---

## Mössingen Agents (`moessingen_agents/`)

FastAPI service exposing 5 AI agent personas for everyday life in Mössingen and the Schwäbische Alb.

**Start the server:**
```bash
cd /root/AI/moessingen_agents
pip install -r requirements.txt
cp .env.example .env  # add your ANTHROPIC_API_KEY
uvicorn app.main:app --reload --port 8000
```

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Returns `{status, agents, provider}` |
| `POST` | `/chat` | Send `{agent, message, language?}`, get `{reply, agent}` |

**Agent IDs:**

| ID | Persona | What it does |
|----|---------|-------------|
| `buergeramt_helper` | BürgeramtHelper | Checklists + document requirements for German bureaucracy. Never invents Behörde details — tells user to verify on official site |
| `german_letter_explainer` | GermanLetterExplainer | Paste a German letter → get summary, deadlines, required actions, and a draft reply |
| `handwerker_finder` | HandwerkerFinder | Writes craftsman inquiry templates. Never invents phone numbers/contacts |
| `event_ideas` | EventIdeas | Weekend ideas for Mössingen / Tübingen / Schwäbische Alb (hikes, museums, family) |
| `small_business_booster` | SmallBusinessBooster | Facebook posts, flyer text, WhatsApp Business templates for local businesses |

**Example request:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"agent": "event_ideas", "message": "Ideen für Sonntag mit Kindern", "language": "de"}'
```

**LLM Provider:** Defaults to Anthropic (`claude-opus-4-6` with adaptive thinking). Set `LLM_PROVIDER=openai` in `.env` to use OpenAI instead.

**Run tests:**
```bash
cd /root/AI/moessingen_agents
pytest
```

**Files:**
```
moessingen_agents/
├── app/
│   ├── main.py          # FastAPI app, /health, /chat
│   ├── config.py        # pydantic-settings (API keys, model names)
│   ├── schemas.py       # ChatRequest, ChatResponse, HealthResponse
│   ├── llm.py           # LLMClient ABC + Anthropic/OpenAI implementations
│   ├── agents/
│   │   ├── base.py      # Agent dataclass
│   │   └── registry.py  # REGISTRY dict with all 5 agents
│   └── prompts/         # One file per agent with SYSTEM_PROMPT string
├── tests/
│   ├── conftest.py      # MockLLMClient + fixtures
│   ├── test_health.py   # /health tests
│   └── test_schemas.py  # Schema + /chat tests
├── .env.example
├── requirements.txt
└── README.md
```

---

## Environment & Setup

**Activate venv:**
```bash
source /root/AI/.venv/bin/activate
```

**Required `.env` variables:**

| Variable | Used by |
|----------|---------|
| `FIREFLIES_API_KEY` | `add_to_fireflies.py` and pipeline scripts |
| `REGISTRANT_NAME` | `register_zoom_meeting.py`, `zoom_to_fireflies.py` |
| `REGISTRANT_EMAIL` | `register_zoom_meeting.py`, `zoom_to_fireflies.py` |

**Google OAuth:**
`credentials.json` and `token.json` must be in the project root for any script that accesses Gmail or Google Calendar. First-time setup requires browser authorization.

**Windows-only tools** (require WSL + Windows):
- `conver_camtasia_files.py` / `export_camtasia_trec.py` — needs AutoHotkey v2 on Windows
- `save_course_as_pdf.py` — output path is a Windows path (`/mnt/c/...`)
- `generate_notebooklm_artifacts.py` — output path is a Windows path

---

## Web Apps — Gradio / Streamlit / FastAPI

Three UI frameworks implemented as learning exercises. All apps live in dedicated folders.

### Gradio Apps (`gradio_apps/`)

```bash
source .venv/bin/activate

# App 1: Simple GPT-4o chat (model picker + system prompt)
python gradio_apps/1_simple_chat.py          # → http://localhost:7860

# App 2: Repo chat (search 12 cached GitHub repos)
python gradio_apps/2_repo_chat.py            # → http://localhost:7861

# App 3: RAG pipeline chat (ChromaDB + upload & index tab)
PYTHONPATH=. python gradio_apps/3_rag_chat.py  # → http://localhost:7862

# All support share=True for a public URL (set in launch() call)
```

### Streamlit App (`streamlit_apps/`)

```bash
# Polished dark sidebar chat — repo mode + simple chat mode
streamlit run streamlit_apps/rag_chat.py --server.address 0.0.0.0 --server.port 8501
# → http://localhost:8501
```

### FastAPI + HTML Web App (`web_app/`)

```bash
# Backend API (streaming SSE + /repos endpoint)
PYTHONPATH=. uvicorn web_app.backend:app --host 0.0.0.0 --port 8000 --reload
# → http://localhost:8000  (serves index.html at /)
```

**Skills saved:**
- `~/.claude/skills/streamlit-app.md` — trigger: "build a streamlit app"
- `~/.claude/skills/web-app-from-scratch.md` — trigger: "build a web app" / "Lovable-style"

---

## Upwork Playwright Scraper

Replaces broken RSS feeds. Uses Playwright + Cloudflare session cookies.

```bash
# Requires in .env: UPWORK_MASTER_TOKEN, UPWORK_XSRF_TOKEN, UPWORK_USER_UID, UPWORK_CF_CLEARANCE
python tools/fetch_upwork_playwright.py --dry-run --query "AI agent" --hours-old 2
python tools/fetch_upwork_playwright.py --query "AI agent" --query "Python developer"
```

**Cookie refresh:** Get fresh `cf_clearance` from Chrome DevTools → Application → Cookies → upwork.com when scraper returns 0 results.

---

## YouTube Playlist Tool

Fetches playlists and videos from YouTube channel via Google OAuth.

```bash
python tools/get_youtube_playlists.py
# Saves to .tmp/youtube_playlists.json
# Uses token_youtube.json (separate from Zoom token.json)
```

**Setup:** Requires YouTube Data API v3 scope in Google Cloud Console + http://localhost:8080 redirect URI.

