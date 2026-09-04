"""Generation of every number that reaches README.md and REPORT.md.

Nothing in those documents is hand-typed. `make report` regenerates the tables
and figures from the committed result JSON, so a stale number cannot survive a
re-run: if the documents change, `git diff` says so.
"""
