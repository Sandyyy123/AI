# AI Agent Ideas & Implementation Tracker

> This file tracks agent ideas from concept to completion. Each agent follows the WAT framework: a workflow (SOP), one or more tools (Python scripts), and the agent logic (me).

---

## How to Add a New Idea

Copy the template below into the **Backlog** section. Fill in what you know — leave the rest blank until implementation.

---

## Implementation Process (Cost-Minimizing)

Follow these steps for every new agent to keep cost and complexity low:

**Step 1 — Define the job clearly**
Write one sentence: "This agent takes X and produces Y." If you can't do this, the idea isn't ready.

**Step 2 — Check existing tools**
Before writing any code, look at `tools/` and this file's "Completed" section. Can you chain existing scripts to solve it?

**Step 3 — Map the steps manually first**
Do the task by hand once. Write down every step. This becomes the workflow SOP in `workflows/`.

**Step 4 — Identify the cheapest execution path**
- Prefer free APIs over paid (e.g., local models over OpenAI where quality allows)
- Prefer one LLM call over many (batch, don't chain unnecessarily)
- Prefer deterministic scripts over LLM calls where possible (parsing, formatting, file ops)
- Use `haiku` or `gpt-4o-mini` for cheap classification/routing; reserve `sonnet`/`gpt-4o` for reasoning

**Step 5 — Build the tool first, workflow second**
Write the Python script in `tools/`. Test it in isolation. Only write the workflow SOP after the tool works.

**Step 6 — Add a `--dry-run` flag**
Every tool that calls a paid API or sends data externally should support `--dry-run` to validate inputs without spending credits.

**Step 7 — Document and update COMMANDS.md**
Add the tool to `COMMANDS.md` so it's discoverable. Update the `Last updated:` date.

---

## Agent Template

```
### [Agent Name]

**Status:** Backlog | In Progress | Completed | Paused

**Job:** One sentence — takes X, produces Y.

**Trigger:** What kicks this off? (manual command, schedule, event, webhook)

**Inputs:**
-

**Outputs / Deliverable:**
-

**Workflow SOP:** `workflows/<filename>.md` (create when starting)

**Tools needed:**
- Existing:
- New to build:

**API / Cost estimate:**
- Service:
- Approx cost per run:

**Notes / Edge cases:**
-
```

---

## Backlog

> Ideas not yet started. Prioritize by value/cost ratio.

---

### WhatsApp Daily Digest

**Status:** Backlog

**Job:** Every 8 hours, fetch all WhatsApp messages received since the last run and send back a concise summary grouped by chat/sender.

**Trigger:** Scheduled — runs every 8 hours (APScheduler or system cron)

**Inputs:**
- WhatsApp message history for the past 8 hours (via Green API or whatsapp-web.js)
- Optional: list of chats to include/exclude

**Outputs / Deliverable:**
- Summary message sent back to yourself on WhatsApp (or saved to a log file)
- Format: grouped by chat, key points per group, total message count

**Workflow SOP:** `workflows/whatsapp_digest.md` (create when starting)

**Tools needed:**
- Existing: none
- New to build:
  - `tools/whatsapp_webhook_receiver.py` — Flask server that receives TimelinesAI webhooks and logs messages to `.tmp/whatsapp_log.jsonl`
  - `tools/fetch_whatsapp_messages.py` — reads log file for the last N hours, groups by chat
  - `tools/summarize_whatsapp.py` — sends grouped messages to Claude Haiku, returns structured summary
  - `tools/send_whatsapp_message.py` — POSTs summary back to your own WhatsApp via TimelinesAI API

**API / Cost estimate:**
- WhatsApp access: TimelinesAI (free trial, then paid) — webhook-based, no polling needed
- Summarization: Claude Haiku (~$0.001 per 8-hour batch of typical chat volume)
- Approx cost per run: near zero

**Implementation order (cost-minimizing):**
1. Sign up for TimelinesAI, connect your WhatsApp account via QR code
2. Get API token from https://app.timelines.ai/integrations/api/ — add `TIMELINESAI_API_TOKEN` and `TIMELINESAI_WHATSAPP_ACCOUNT` to `.env`
3. Run `tools/whatsapp_webhook_receiver.py` (Flask server on port 5055)
4. Expose with ngrok: `ngrok http 5055` → copy the https URL
5. Set webhook in TimelinesAI: https://app.timelines.ai/integrations/webhooks_v2/ → paste URL → event: `chat:incoming:new` → aggregation: Don't aggregate
6. Test: send yourself a WhatsApp message and verify it appears in `.tmp/whatsapp_log.jsonl`
7. Build `summarize_whatsapp.py` — one Haiku call, structured prompt grouping by sender
8. Build `send_whatsapp_message.py` — POST to TimelinesAI `/messages` endpoint
9. Wire into `tools/whatsapp_digest.py` pipeline
10. Add cron: `0 */8 * * * cd /root/AI && source .venv/bin/activate && python tools/whatsapp_digest.py`

**Notes / Edge cases:**
- Webhook receiver must stay running (use tmux: `tmux new -s whatsapp_webhook`)
- ngrok free tier resets URL on restart — update TimelinesAI webhook URL each time, or use a paid ngrok plan / VPS
- Media messages (images, voice notes) — logged as type, not content
- Messages you sent are filtered out — only received messages are logged and summarized
- TimelinesAI full API reference: https://app.timelines.ai/integrations/api/redoc

---

### Scheduled Reminder Agent

**Status:** Backlog

**Job:** Takes a reminder message and schedule, and delivers it to the user at the specified time(s).

**Trigger:** Scheduled — cron or APScheduler based on user-defined time/recurrence

**Inputs:**
- Reminder message (text)
- Schedule (datetime or cron expression)
- Delivery channel (WhatsApp, desktop notification, email, etc.)

**Outputs / Deliverable:**
- Reminder delivered to the user via configured channel at the scheduled time

**Workflow SOP:** `workflows/scheduled_reminders.md` (create when starting)

**Tools needed:**
- Existing: `tools/send_whatsapp_message.py` (if WhatsApp delivery)
- New to build:
  - `tools/schedule_reminder.py` — registers a reminder (message + schedule) and persists it
  - `tools/run_reminder_scheduler.py` — daemon that checks due reminders and dispatches them

**API / Cost estimate:**
- Delivery via WhatsApp: TimelinesAI (see WhatsApp Daily Digest)
- Delivery via desktop: free (notify-send / plyer)
- Approx cost per run: near zero

**Notes / Edge cases:**
- Persist reminders to a file (e.g., `.tmp/reminders.json`) so they survive restarts
- Support one-time and recurring reminders
- Timezone handling — store all times in UTC, display in user local time

---

### Subscription Tracker

**Status:** Backlog

**Job:** Scans email inbox for payment/subscription receipts and produces a list of active subscriptions with service name, amount, and billing date.

**Trigger:** Manual or scheduled monthly run

**Inputs:**
- Access to email inbox (Gmail via OAuth or IMAP)
- Optional: date range to scan

**Outputs / Deliverable:**
- Structured list of subscriptions: service name, amount, currency, billing frequency, last payment date
- Saved to Google Sheets or exported as CSV

**Workflow SOP:** `workflows/subscription_tracker.md` (create when starting)

**Tools needed:**
- Existing: Google OAuth (credentials.json, token.json)
- New to build:
  - `tools/fetch_emails.py` — fetches emails matching keywords (receipt, invoice, payment, subscription) via Gmail API
  - `tools/extract_subscriptions.py` — sends email bodies to Claude Haiku to extract structured subscription data
  - `tools/export_subscriptions.py` — writes results to Google Sheets or CSV

**API / Cost estimate:**
- Gmail API: free
- Extraction: Claude Haiku (~$0.002 per batch of typical inbox)
- Approx cost per run: near zero

**Notes / Edge cases:**
- Filter by sender keywords: "receipt", "invoice", "payment confirmed", "your subscription"
- Deduplicate — same service may appear multiple times (monthly charges)
- Handle multi-currency amounts
- Flag subscriptions not charged in 60+ days as potentially cancelled

---

### YouTube Production Agent

**Status:** Backlog

**Job:** Takes a video file or topic and produces all the assets needed to publish a YouTube video — title, description, tags, chapters, thumbnail brief, and upload-ready metadata.

**Trigger:** Manual — run when a video is ready for publishing

**Inputs:**
- Video file or transcript (`.mp4`, `.txt`, or `.srt`)
- Optional: target audience, tone, keywords

**Outputs / Deliverable:**
- YouTube title (A/B options)
- Description with chapters, keywords, and CTA
- Tags list (15–20)
- Pinned comment draft
- Thumbnail brief (visual concept + text overlay suggestion)
- All saved to a Google Doc or Markdown file

**Workflow SOP:** `workflows/youtube_production.md` (create when starting)

**Tools needed:**
- Existing: none
- New to build:
  - `tools/transcribe_video.py` — extracts audio and transcribes via Whisper (local) or AssemblyAI
  - `tools/generate_youtube_assets.py` — sends transcript to Claude Sonnet, returns structured metadata
  - `tools/export_youtube_doc.py` — writes all assets to a Google Doc or Markdown file

**API / Cost estimate:**
- Transcription: Whisper local (free) or AssemblyAI (~$0.01/min)
- Asset generation: Claude Sonnet (~$0.01–0.05 per video)
- Approx cost per run: <$0.10 for a typical video

**Notes / Edge cases:**
- If no transcript provided, auto-transcribe from video file
- Chapters extracted from transcript timestamps — requires reasonably accurate timecodes
- Title A/B options: one curiosity-driven, one keyword-driven
- Thumbnail brief should describe visual layout, not just text

---

### LinkedIn Post Agent

**Status:** Backlog

**Job:** Takes a topic, article, or transcript and produces a ready-to-publish LinkedIn post with hooks, body, and call-to-action.

**Trigger:** Manual — run with a topic or source content

**Inputs:**
- Topic, URL, transcript, or raw notes
- Optional: tone (personal story, thought leadership, educational), target audience

**Outputs / Deliverable:**
- 2–3 LinkedIn post variants (different hooks/angles)
- Each post: hook line, body (3–5 short paragraphs), CTA, hashtags
- Saved to a Markdown file or Google Doc

**Workflow SOP:** `workflows/linkedin_post.md` (create when starting)

**Tools needed:**
- Existing: none
- New to build:
  - `tools/fetch_source_content.py` — fetches article or URL content if input is a link
  - `tools/generate_linkedin_posts.py` — sends content to Claude Sonnet, returns 2–3 post variants

**API / Cost estimate:**
- Content fetch: free (requests + BeautifulSoup)
- Generation: Claude Sonnet (~$0.01 per run)
- Approx cost per run: near zero

**Notes / Edge cases:**
- LinkedIn character limit: 3,000 characters — enforce in output
- Hook line is critical — generate 3 hook options per post variant
- Avoid corporate buzzwords — flag if detected
- Format for LinkedIn readability: short paragraphs, line breaks, no walls of text

---

### WAT Web Dashboard

**Status:** Backlog

**Job:** A Lovable-built React web app that surfaces the WAT system as a UI — view and manage the agent backlog, trigger tool runs, and browse outputs without touching the terminal.

**Trigger:** Always-on web app; tool runs triggered manually via UI buttons

**Inputs:**
- Supabase DB synced from local WAT files (AGENTS.md, tool outputs)
- FastAPI bridge server running locally to execute Python tools on demand

**Outputs / Deliverable:**
- Live web dashboard accessible from any device
- Kanban board of agents (Backlog / In Progress / Completed)
- Output viewer for LinkedIn posts, YouTube assets, subscription lists, etc.
- Trigger buttons to run tools remotely

**Workflow SOP:** `workflows/wat_web_dashboard.md` (create when starting)

**Tools needed:**
- Existing: all existing `tools/` scripts (exposed via FastAPI)
- New to build:
  - `tools/sync_agents_to_supabase.py` — parses AGENTS.md and upserts agent records to Supabase
  - `tools/api_server.py` — FastAPI server that wraps tool scripts as HTTP endpoints
- External: Lovable (React UI builder), Supabase (DB + auth)

**API / Cost estimate:**
- Supabase: free tier sufficient
- Lovable: free tier or ~$20/mo for pro
- FastAPI server: runs locally, exposed via ngrok or VPS

**Notes / Edge cases:**
- FastAPI bridge must stay running to allow remote tool triggers (use tmux)
- ngrok free tier resets URL on restart — use a VPS or paid ngrok for persistent access
- Start with read-only dashboard (sync + view), add trigger buttons after
- Auth: Supabase Auth or simple API key to prevent public access

---

### Upwork Automated Job Applier

**Status:** Backlog

**Job:** Monitors Upwork for new job postings matching your AI/tech skills, scores each for fit, and generates a tailored proposal — ready to send or auto-submitted.

**Trigger:** Scheduled — runs every 2–4 hours during active hours to catch fresh postings before they fill

**Inputs:**
- Upwork profile / skills summary (AI engineering, RAG, agents, Python, automation)
- Target job categories: AI Development, Python, Automation, LLM Integration, Web Scraping
- Budget filters: min $25/hr for hourly, min $200 for fixed-price
- Dealbreakers: low-budget jobs (<$15/hr), vague scope, "ongoing for years" without clear deliverable

**Outputs / Deliverable:**
- Ranked list of new job postings with fit score (0–10) and recommended bid price
- Tailored Upwork proposal per job (opening hook + relevant experience + CTA)
- Application log in Google Sheets (job title → applied → response → contract)

**Workflow SOP:** `workflows/upwork_job_applier.md` (create when starting)

**Tools needed:**
- Existing: none
- New to build:
  - `tools/fetch_upwork_jobs.py` — polls Upwork RSS feed or API for new postings by keyword/category
  - `tools/score_upwork_job.py` — sends job description + profile to Claude Haiku, returns fit score + red flags
  - `tools/generate_upwork_proposal.py` — sends job + profile to Claude Sonnet, returns tailored proposal
  - `tools/submit_upwork_proposal.py` — (optional) uses Playwright to auto-submit proposal via browser automation
  - `tools/track_upwork_applications.py` — logs job + proposal + status to Google Sheets

**Proposal Strategy:**
- First 2 lines are everything — open with the client's problem, not your credentials
- Keep proposals under 150 words — clients skim, not read
- Always include one specific observation about their job post ("I noticed you mentioned X — I've solved this by...")
- End with a single low-friction CTA: "Want me to share a quick example of how I'd approach this?"
- Attach relevant portfolio samples (RAG demo, agent workflow) as proof of work

**Targeting Strategy:**
- **Sweet spot**: $500–$5,000 fixed-price AI projects with clear scope
- **Best categories**: AI/ML Development, Python, Automation & Scripting, Data Extraction, Chatbot Development
- **Client signals to favor**: payment verified, 4.5+ rating, clear job description, previous hires
- **Avoid**: "looking for cheapest option", no budget listed, 50+ proposals already submitted
- **Rising Talent / Top Rated**: aim for this badge — prioritize early wins at competitive rates to build review count

**API / Cost estimate:**
- Job fetching: Upwork RSS feed (free) or Upwork API (requires partner approval — use RSS first)
- Fit scoring: Claude Haiku (~$0.001 per job)
- Proposal generation: Claude Sonnet (~$0.01 per proposal)
- Approx cost per run: <$0.05 for a batch of 10 jobs

**Notes / Edge cases:**
- Upwork RSS feeds per category: `https://www.upwork.com/ab/feed/jobs/rss?q=AI+agent&sort=recency`
- Auto-submission via Playwright is risky — Upwork detects bots; safer to generate proposal and copy-paste manually
- Connects (Upwork's credits) cost ~$0.15 each — don't waste on low-fit jobs; only submit to 7+/10 matches
- Track which proposal styles get responses — A/B test opening hooks
- Use Upwork's "Saved Searches" + email alerts as a backup trigger alongside the script

---

### AI Skills Learning Roadmap Agent

**Status:** Backlog

**Job:** Builds and tracks a structured self-study plan for mastering Claude Code, Kimi Moonshot, and OpenClaw — delivering daily learning tasks, project prompts, and progress tracking.

**Trigger:** Manual — run daily to get today's learning task + track completion

**Inputs:**
- Current skill level per tool (beginner / intermediate / advanced)
- Available study time per day (e.g., 30min, 1h, 2h)
- Learning goal (e.g., "build agents", "use for freelance projects", "teach others")
- Optional: completed topics so far

**Outputs / Deliverable:**
- Daily learning task with explanation, exercise, and what to build
- Weekly skill summary: what was covered, what to review, what's next
- Progress tracker in Google Sheets (topic → studied → practiced → applied)
- Project ideas to apply each tool in real WAT agents

**Workflow SOP:** `workflows/ai_skills_learning.md` (create when starting)

**Tools needed:**
- Existing: none
- New to build:
  - `tools/generate_daily_lesson.py` — generates today's focused learning task based on curriculum + progress
  - `tools/track_learning_progress.py` — logs completed topics to Google Sheets

---

**Curriculum: Claude Code**
> Goal: Master the Claude Code CLI to build, debug, and orchestrate AI agents directly from the terminal

- **Week 1 — Foundations**: Install + authenticate, basic chat and file editing, reading codebases with Claude
- **Week 2 — Agents**: Multi-step tasks, tool use (Bash, Read, Write, Edit, Grep, Glob), memory files
- **Week 3 — WAT Integration**: Build a full WAT workflow using Claude Code end-to-end — from SOP to tool to output
- **Week 4 — Advanced**: Custom slash commands, CLAUDE.md project instructions, hooks, parallel tool calls
- **Practice project**: Use Claude Code to build and test one tool from AGENTS.md backlog each week

**Resources:**
- Official docs: https://docs.anthropic.com/claude-code
- CLAUDE.md in this repo — already using the framework

---

**Curriculum: Kimi Moonshot**
> Goal: Understand Kimi k1.5 / k2 capabilities and use Moonshot API as a cost-effective alternative for long-context and reasoning tasks

- **Week 1 — What is Kimi**: Moonshot AI background, k1.5 vs k2 models, strengths (long context up to 1M tokens, multilingual, reasoning)
- **Week 2 — API Access**: Sign up at platform.moonshot.cn, get API key, run first completion via Python SDK (compatible with OpenAI SDK format)
- **Week 3 — Use Cases**: Long document analysis, Chinese/multilingual content, cost comparison vs Claude/GPT
- **Week 4 — Integration**: Drop Kimi into a WAT tool as an alternative LLM backend — test quality vs cost
- **Practice project**: Use Kimi to process a long German document (e.g., Mietvertrag, Behördenschreiben) and extract key facts

**Resources:**
- API docs: https://platform.moonshot.cn/docs
- Compatible with OpenAI Python SDK — just swap base_url and model name

---

**Curriculum: OpenClaw (Open-Source Claude Alternatives)**
> Goal: Learn to use and self-host open-source models that replicate Claude-like agent behavior — for privacy, cost control, and offline use

- **Week 1 — Landscape**: Understand the open-source LLM ecosystem — Llama 3, Mistral, Qwen, DeepSeek, Phi-4
- **Week 2 — Local Setup**: Install Ollama (easiest local runner), pull a model (`ollama pull llama3`), run basic completions
- **Week 3 — Agent Frameworks**: LangChain + local models, LlamaIndex, or bare API calls — replicate a WAT tool using a local model
- **Week 4 — Evaluation**: Compare output quality vs Claude Haiku for specific tasks (summarization, classification, extraction)
- **Practice project**: Replace one Claude Haiku call in an existing tool with a local Ollama model — measure quality and speed

**Resources:**
- Ollama: https://ollama.com (local model runner)
- Open WebUI: browser-based UI for Ollama
- Model comparison: https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard

---

**Learning Principles:**
- Learn by building — every week ends with a working project, not just notes
- Apply immediately to WAT — if you can't use it in a real tool this week, revisit the lesson
- Teach to retain — after each module, write a short LinkedIn post or internal summary explaining what you learned
- Timebox — 1 focused hour per day beats 4 scattered hours
- Track blockers — log what confused you; these become future workflow improvements

**API / Cost estimate:**
- Lesson generation: Claude Haiku (~$0.001/day)
- Kimi API: very low cost (~$0.001/1K tokens for k1.5)
- Ollama: free (runs locally)
- Approx cost per run: near zero

**Notes / Edge cases:**
- Start with Claude Code — it directly accelerates everything else in WAT
- Kimi is especially useful for German/Arabic content processing given multilingual strength
- "OpenClaw" interpreted as open-source Claude-like tools — update this section if you meant a specific product
- Don't try to learn all three simultaneously — rotate weekly focus

---

### Wife's German Job Search Agent

**Status:** Backlog

**Job:** Finds relevant job openings in Germany for a journalist-trained, social-sector-experienced candidate with intermediate German, and generates tailored application materials in German per listing.

**Trigger:** Manual — run daily during active job search

**Inputs:**
- Candidate profile: journalism training, social work experience (voluntary/freelance), intermediate German (B1/B2)
- Target regions: Germany (local area + remote/hybrid roles)
- Preferred sectors: social work, NGOs, community organizations, communications, PR, media, Redaktion
- Dealbreakers: roles requiring C1+ German without growth path, full-time only if needed

**Outputs / Deliverable:**
- Daily ranked list of matching jobs with fit score and language-barrier risk rating
- Tailored Bewerbungsanschreiben (cover letter in German) per application
- Lebenslauf (CV) adaptation notes per role
- LinkedIn / XING message to hiring manager or Ansprechpartner
- Application tracker in Google Sheets

**Workflow SOP:** `workflows/wife_job_search.md` (create when starting)

**Tools needed:**
- Existing: `tools/score_job_fit.py`, `tools/generate_cover_letter.py`, `tools/track_applications.py` (adapt from Job Search agent)
- New to build:
  - `tools/scrape_german_job_boards.py` — fetches listings from Indeed.de, StepStone, Arbeitsagentur (BA), Idealist.de (NGO/social), Jobbörse, XING Jobs
  - `tools/generate_german_cover_letter.py` — generates formal German Bewerbungsanschreiben tailored to role + candidate profile

**Target Roles (Priority Order):**
1. **Sozialarbeiter / Sozialpädagoge** — NGOs, Caritas, Diakonie, AWO, DRK (social work experience counts even if voluntary)
2. **Kommunikation / PR / Öffentlichkeitsarbeit** — NGOs, Vereine, public institutions (journalism training is a strong fit)
3. **Redakteur / Texter** — local newspapers, online media, corporate communications (lower German bar for writing roles if content is reviewed)
4. **Beratung / Case Management** — refugee support, integration services, Jobcenter partners (high demand, language support often provided)
5. **Ehrenamtskoordinator** — coordinating volunteers at NGOs (her voluntary background is directly relevant)

**Key Unlocks:**
- **Anerkennung**: Check if journalism or social work qualifications can be formally recognized (anabin.kmk.org + Anerkennungsberatung)
- **Bundesagentur für Arbeit**: Register as jobseeking (Arbeitssuchend melden) — unlocks Beratung, subsidized German courses, integration services
- **Qualifizierungsmaßnahmen**: BA may fund a short Auffrischungskurs in social work or a German course to bridge to C1
- **Praktikum / Hospitanz**: Short unpaid trial at a local NGO builds German references fast
- **Netzwerk**: Volunteer at Caritas or DRK first — many hires come from within the volunteer pool

**API / Cost estimate:**
- Job scraping: free (Indeed.de RSS, BA Jobbörse API is public)
- Fit scoring + cover letter: Claude Haiku + Sonnet (~$0.02 per application)
- Approx cost per run: <$0.05 for a daily batch

**Notes / Edge cases:**
- German Bewerbung format is strict: formal salutation, no humor, structured paragraphs, sign off with "Mit freundlichen Grüßen"
- Many social sector roles accept B2 German — do not auto-filter these out
- Arbeitsagentur Jobbörse has an API (no auth needed): `https://rest.arbeitsagentur.de/jobboerse/jobsuche-service/pc/v4/jobs`
- XING is more relevant than LinkedIn for German SME and NGO roles — prioritize it
- Flag roles at Träger (Caritas, AWO, Diakonie, DRK, Paritätischer) — these are the main social work employers in Germany

---

### Job Search & Application Agent

**Status:** Backlog

**Job:** Takes a target role and location, finds relevant job postings, scores them against a profile, and generates tailored application materials (CV cover letter, LinkedIn message to hiring manager).

**Trigger:** Manual — run daily or on demand during active job search

**Inputs:**
- Target role(s) (e.g., "AI Engineer", "ML Product Manager", "AI Trainer")
- Target locations (e.g., remote, Germany, EU, global)
- CV / LinkedIn profile (text or PDF)
- Optional: salary range, preferred industries, dealbreakers

**Outputs / Deliverable:**
- Ranked list of matching job postings with fit score and gap analysis
- Tailored cover letter per application
- LinkedIn connection message to hiring manager or recruiter
- Application tracker in Google Sheets (job → applied → interview → offer)

**Workflow SOP:** `workflows/job_search_application.md` (create when starting)

**Tools needed:**
- Existing: none
- New to build:
  - `tools/scrape_job_boards.py` — fetches listings from LinkedIn Jobs, Indeed, Wellfound, Remotive (filtered by role + location)
  - `tools/score_job_fit.py` — sends job description + CV to Claude Haiku, returns fit score (0–10) + missing skills
  - `tools/generate_cover_letter.py` — sends job description + CV to Claude Sonnet, returns tailored cover letter
  - `tools/generate_recruiter_message.py` — generates short LinkedIn outreach message to hiring manager
  - `tools/track_applications.py` — upserts job + status into Google Sheets tracker

**Strategy:**
- Focus on AI-specific job boards first: Wellfound (startups), Remotive (remote), LinkedIn (corporate)
- Score every listing before applying — only apply to 7+/10 matches to keep quality high
- Always send a recruiter/hiring manager LinkedIn message within 24h of applying
- Tailor cover letter to the specific pain the company is solving — not generic summaries
- Track every application; follow up after 7 days if no response

**API / Cost estimate:**
- Job scraping: free (requests + BeautifulSoup or Playwright for JS sites)
- Fit scoring: Claude Haiku (~$0.001 per listing)
- Cover letter: Claude Sonnet (~$0.01 per letter)
- Approx cost per run: <$0.10 for a batch of 20 listings

**Notes / Edge cases:**
- LinkedIn scraping may require login — use their Jobs RSS feed or a third-party API (RapidAPI LinkedIn Jobs)
- Deduplicate listings across boards by normalizing company name + job title
- Store raw listings in `.tmp/job_listings.jsonl` — regenerate daily
- Flag roles that require visa sponsorship if targeting outside EU

---

### AI Papers to Social Media Publisher

**Status:** Backlog

**Job:** Takes AI research papers (PDF or arXiv link), converts them to an HTML summary, and publishes platform-optimized posts to LinkedIn, Twitter/X, Facebook, and Instagram.

**Trigger:** Manual — run when a new paper is worth sharing, or scheduled daily to pick from a curated feed

**Inputs:**
- Paper source: PDF file path, arXiv URL, or Hugging Face paper link
- Target platforms: LinkedIn, Twitter/X, Facebook, Instagram (configurable per run)
- Optional: tone (technical, accessible, hype), audience notes

**Outputs / Deliverable:**
- Clean HTML summary of the paper (key findings, method, results, implications)
- Platform-optimized posts:
  - LinkedIn: 1,500–2,000 char thought leadership post with hook, breakdown, takeaway, hashtags
  - Twitter/X: thread (8–12 tweets) starting with a hook, each tweet one key insight
  - Facebook: 300–500 char accessible summary with image caption suggestion
  - Instagram: caption (2,200 char max) + slide content brief for carousel (5–7 slides)
- All outputs saved to `.tmp/paper_posts_YYYY-MM-DD/`

**Workflow SOP:** `workflows/ai_papers_social_publisher.md` (create when starting)

**Tools needed:**
- Existing: none
- New to build:
  - `tools/fetch_paper.py` — downloads PDF from arXiv/URL or reads local file; extracts full text
  - `tools/convert_paper_to_html.py` — sends extracted text to Claude Sonnet, returns structured HTML summary (abstract, method, results, implications)
  - `tools/generate_social_posts.py` — takes HTML summary, generates platform-specific posts per channel
  - `tools/publish_to_social.py` — (optional) posts to each platform via API (LinkedIn API, Twitter API v2, Facebook Graph API, Instagram Graph API)

**API / Cost estimate:**
- PDF extraction: free (PyMuPDF / pdfplumber)
- arXiv fetch: free
- HTML summary + post generation: Claude Sonnet (~$0.03–0.08 per paper across all platforms)
- Publishing APIs: LinkedIn, Twitter, Facebook, Instagram — all have free tiers with rate limits
- Approx cost per run: <$0.10 per paper

**Notes / Edge cases:**
- Twitter character limit: 280 per tweet — enforce strictly in thread generator
- LinkedIn: no direct image upload via API without media asset upload step — note in workflow
- Instagram requires an image — generate a slide brief + prompt the user to create the visual, or use a template
- Facebook Graph API requires page token (not personal profile) — document setup steps
- arXiv rate limit: 3 requests/sec — add `time.sleep(0.5)` between fetches
- For paywalled papers: fall back to abstract + title only, note limitation in output
- Add `--dry-run` flag to preview posts without publishing

---

### AI Paper → NotebookLM Audio/Video Publisher

**Status:** Backlog

**Job:** Takes an AI research paper, generates a NotebookLM podcast audio summary, converts it to a video (with slides or visuals), and publishes the video to YouTube/Instagram Reels/TikTok.

**Trigger:** Manual — run per paper, or scheduled daily from a curated arXiv feed

**Inputs:**
- Paper source: PDF, arXiv URL, or Hugging Face link
- Optional: target audience (researchers, builders, general), output platform(s)

**Outputs / Deliverable:**
- NotebookLM notebook with paper as source
- AI-generated podcast audio (MP3) — 5–10 min conversational summary via NotebookLM
- Slide deck (HTML or image frames) generated from the paper's key points
- Final video: audio + slides combined (MP4) — ready to upload
- Platform-optimized title, description, tags for YouTube
- Short-form cut (60s) for Instagram Reels / TikTok

**Workflow SOP:** `workflows/ai_paper_video_publisher.md` (create when starting)

**Tools needed:**
- Existing: `tools/fetch_paper.py` (from AI Papers to Social Media agent), NotebookLM skill
- New to build:
  - `tools/create_notebooklm_podcast.py` — creates NotebookLM notebook, adds paper as source, triggers Audio Overview generation, downloads MP3
  - `tools/generate_paper_slides.py` — sends paper summary to Claude Sonnet, returns slide content (title + 5–7 bullet slides); renders to HTML or PNG frames
  - `tools/combine_audio_slides.py` — uses FFmpeg to sync audio with slide frames into MP4
  - `tools/upload_to_youtube.py` — uploads MP4 + metadata via YouTube Data API v3
  - `tools/cut_short_form.py` — trims first 60s of audio+slides for Reels/TikTok cut

**API / Cost estimate:**
- NotebookLM: free (Google account required); audio generation is free
- Slide generation: Claude Sonnet (~$0.02 per paper)
- FFmpeg: free (local)
- YouTube upload: free (YouTube Data API v3, OAuth)
- Approx cost per run: <$0.05 per paper

**Notes / Edge cases:**
- NotebookLM Audio Overview takes 2–5 min to generate — poll for completion before downloading
- Audio download from NotebookLM requires automation (Playwright) — no official download API yet
- FFmpeg slide timing: divide audio duration equally across slides, or use word-count heuristic per slide
- YouTube daily upload quota: 6 uploads/day on unverified accounts — verify account for higher limits
- Short-form cut: extract the most compelling 60s (intro hook) — use Claude to identify the best segment from transcript
- Chain this agent after the **AI Papers to Social Media Publisher** — same paper, both text posts + video in one run

---

### AI Network Builder (Local + International)

**Status:** Backlog

**Job:** Takes a target market (local or international) and executes an outreach and relationship-building strategy to grow a network of clients, partners, and collaborators for AI training and product delivery.

**Trigger:** Manual — run weekly or on demand to advance network-building campaigns

**Inputs:**
- Target segment (e.g., local SMEs in Mössingen/Tübingen, international tech founders, HR managers)
- Offer type (AI training workshop, custom AI product build, consulting)
- Outreach channel (LinkedIn, email, WhatsApp, events)
- Optional: existing contact list or CRM export

**Outputs / Deliverable:**
- Prioritized list of target contacts with LinkedIn/email and personalization notes
- Outreach message variants per channel (LinkedIn DM, cold email, event follow-up)
- Follow-up sequence (3-touch: intro → value → CTA)
- Weekly pipeline tracker in Google Sheets (contact → reached → replied → meeting booked → deal)

**Workflow SOP:** `workflows/ai_network_builder.md` (create when starting)

**Tools needed:**
- Existing: none
- New to build:
  - `tools/find_local_leads.py` — scrapes or queries local business directories (IHK, LinkedIn, Google Maps) for SMEs in target region
  - `tools/find_international_leads.py` — searches LinkedIn or Apollo.io for target profiles (job title + industry + location filters)
  - `tools/generate_outreach_messages.py` — sends contact profile to Claude Sonnet, returns personalized DM/email variants
  - `tools/track_pipeline.py` — upserts contact + status into Google Sheets pipeline tracker

**Strategy — Local Network (Germany/Mössingen/Tübingen):**
- Target: Mittelstand companies (50–500 employees), IHK members, local Handwerk with digitization pressure
- Channels: IHK events, LinkedIn (German), in-person workshops at co-working spaces (Tübingen, Reutlingen)
- Offer: half-day AI literacy workshops + follow-on automation audits
- Entry point: free workshop or lunch-and-learn to build trust first
- Partners to cultivate: local IT consultancies, Steuerberater, Unternehmensberater who can refer

**Strategy — International Network:**
- Target: English-speaking founders and team leads in SaaS, e-commerce, professional services (EU, US, MENA)
- Channels: LinkedIn outreach, Twitter/X, AI-focused Slack/Discord communities, async video (Loom)
- Offer: AI product scoping + build (RAG pipelines, agents, automation), team training programs
- Positioning: "European AI engineer with hands-on build experience" — credibility via portfolio + case studies
- Entry point: free audit call or short async Loom teardown of their current workflow

**API / Cost estimate:**
- Lead sourcing: Apollo.io free tier (50 leads/mo) or LinkedIn Sales Navigator (~$80/mo)
- Message generation: Claude Haiku (~$0.001 per contact)
- Pipeline tracking: Google Sheets (free)
- Approx cost per run: near zero (excluding LinkedIn subscription if used)

**Notes / Edge cases:**
- Personalization is the key differentiator — never send generic blasts
- Track reply rates per message variant and A/B test hooks
- Local outreach converts faster but smaller deals — international for larger, longer-cycle deals
- Build in public: share learnings on LinkedIn to attract inbound alongside outbound
- CRM can start as a Google Sheet; migrate to Notion or Airtable when volume grows

---

## In Progress

> Actively being built.

---

### YouTube AI Agent Learning Tracker

**Status:** In Progress

**Job:** Fetches the latest YouTube videos on AI agents, extracts their transcripts, summarizes key insights and tool recommendations, and saves a daily learning digest.

**Trigger:** Manual — run daily to stay current with what's being built and taught in the AI agent space

**Inputs:**
- Search keywords (default: "AI agents", "LangChain agents", "LLM automation", "AI workflow")
- Max videos per run (default: 10)
- Optional: specific YouTube channel URLs to monitor

**Outputs / Deliverable:**
- Daily markdown digest saved to `.tmp/ai_learning_digest_YYYY-MM-DD.md`
- Per-video summary: key insights, tools/frameworks mentioned, actionable techniques
- Running `tools_mentioned.json` tracking which tools appear most across videos
- Optional: Google Sheets log of videos watched + insights

**Workflow SOP:** `workflows/youtube_ai_tracker.md`

**Tools needed (built):**
- `tools/fetch_youtube_ai_videos.py` — searches YouTube RSS for latest AI agent videos
- `tools/extract_youtube_transcript.py` — downloads transcript for each video
- `tools/summarize_video_insights.py` — sends transcript to Claude Haiku, extracts insights + tools
- `tools/save_learning_digest.py` — compiles daily markdown digest + updates tools frequency tracker

**API / Cost estimate:**
- YouTube transcript: free (`youtube-transcript-api`)
- YouTube search: free (RSS/Atom feed)
- Summarization: Claude Haiku (~$0.002 per video)
- Approx cost per run: <$0.02 for 10 videos

---

## Completed

> Finished agents. Reference these before building anything new.

---

## Paused / Shelved

> Good ideas, but blocked or deprioritized. Keep for later.
