# /root/AI/scripts/utils.py
import re

def model_slug(s: str) -> str:
    """
    Converts a string into a URL-friendly slug.
    Removes non-alphanumeric characters and converts to lowercase.
    """
    if not isinstance(s, str):
        s = str(s) # Ensure input is a string
    s = s.lower()
    s = re.sub(r'[^a-z0-9]+', '-', s) # Replace non-alphanumeric with hyphens
    s = s.strip('-') # Remove leading/trailing hyphens
    return s
