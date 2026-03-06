"""
record_meeting.py — Add a Zoom meeting to Fireflies for recording.

Usage:
  python record_meeting.py --date 2026-03-08 --organizer "GEF C4"
  python record_meeting.py --date 2026-03-08 --organizer "GEF C4" --time 14:00
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tools"))

from zoom_to_fireflies import main

if __name__ == "__main__":
    main()


# c