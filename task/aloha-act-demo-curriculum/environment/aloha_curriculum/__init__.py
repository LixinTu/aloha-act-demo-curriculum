"""Task-owned runtime for the aloha-act-demo-curriculum OpenRSI task.

Candidates authored by the research agent are *data* (a JSON curriculum policy and an
optional CSV of per-frame scores). This package validates them, compiles them into a
fixed-length exposure schedule, trains ACT with a fixed recipe, and evaluates the policy
in gym-aloha. No candidate content is ever executed as code.
"""

__version__ = "0.1.0"
