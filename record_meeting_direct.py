"""
record_meeting_direct.py — Record a meeting that doesn't require Zoom registration.
Gets the Zoom join link directly from your Gmail invite email.

Usage:
  python record_meeting_direct.py --date 2026-03-08 --organizer "GEF C4"
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tools"))

from zoom_to_fireflies_direct import main

if __name__ == "__main__":
    main()


# cd /root/AI && source .venv/bin/activate && python record_meeting_direct.py --date 2026-03-08 --organizer "GEF C4"
