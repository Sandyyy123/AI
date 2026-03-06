"""
save_course_pdfs.py — Download all dailydoseofds.com course lessons as PDFs.

Usage:
  python save_course_pdfs.py                        # all courses
  python save_course_pdfs.py --course "MCP"         # one course only
  python save_course_pdfs.py --course "RAG"
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tools"))

from save_course_as_pdf import main

if __name__ == "__main__":
    main()
