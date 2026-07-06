# Rules for Claude in this repo

## Git
- **NEVER run `git commit` unless the user's latest message explicitly says to commit.**
  Implementing a change, verifying it works, or the user approving the *result* is NOT
  permission to commit. Finish the work, then ask "Ready to commit?" and wait for an
  explicit yes. This rule has been violated repeatedly — treat it as a hard stop.
- Commit on the currently checked-out branch (e.g. `master`); do not auto-create
  feature branches.
- Never `git restore` / discard uncommitted changes without asking first.

## Data
- Never run a bare `update-data` (full-universe refetch). Use `evaluate` or
  `update-data -s <SYM>` for specific symbols.
- `reports/` is gitignored output-only — never put scripts there; one-off analysis
  scripts go in `scripts/`.
