# Contributing

Thanks for considering a contribution. Please read this before opening a PR
or filing a substantive issue.

## Setup

See the [README "Setup" section](README.md#setup-one-time) for venv
creation and dependency install. After that, run `pytest` to confirm the
test suite passes (98 tests as of writing) before making changes.

## DCO sign-off — required on every commit

Every commit you submit must carry a `Signed-off-by` trailer. This is the
[Developer Certificate of Origin](https://developercertificate.org/) — the
same lightweight asserttion the Linux kernel uses. By signing off, you state
that you have the right to submit the change under the project's license
(see below) and that the change is your own work or otherwise covered by an
appropriate open-source license.

Add it automatically with `git commit -s`:

```
git commit -s -m "your commit message"
```

That appends a line like:

```
Signed-off-by: Your Name <your.email@example.com>
```

PRs whose commits lack a sign-off will be asked to amend / rebase before
merge. To retroactively add a sign-off:

```
git commit --amend -s --no-edit            # last commit
git rebase --signoff <upstream-branch>     # all commits on this branch
```

## License of contributions

By submitting a contribution, you agree that:

1. **It is licensed under AGPL-3.0-or-later** — the same license that
   covers the rest of this project (see [`LICENSE`](LICENSE)).
2. **The maintainer may, at their discretion, request a separate
   Contributor License Agreement (CLA) for substantial contributions** if
   a future commercial offering requires re-licensing rights. You are free
   to decline; in that case the contribution will simply remain
   AGPL-3.0-only and the maintainer may choose to re-implement the same
   functionality independently for any commercial branch.
3. **You comply with vnstock's non-commercial license** at runtime, the
   same as any user of this project (see [`NOTICE`](NOTICE)).

Most fixes — typos, doc edits, single-bug patches — never need anything
beyond the DCO. The CLA path is only relevant for sizeable architectural
contributions.

## Pull-request workflow

1. **Fork** the repo on GitHub.
2. **Branch** off `main` (or whatever branch the maintainer has marked as
   the default). Use a descriptive branch name (`fix-rate-limiter-deadlock`,
   not `patch-1`).
3. **Make changes** with a clear, small focus. One feature / one bug per PR.
4. **Add or update tests** for behavioral changes. The bar is the existing
   coverage style — small, fast, no network, deterministic.
5. **Run `pytest`** locally and confirm the full suite passes.
6. **Commit with `-s`** so the DCO sign-off is in place.
7. **Open a PR** targeting `main`. Describe the user-visible change in the
   PR body, and reference any related issue.
8. **Expect review.** Maintainer feedback is usually inline; please address
   it via additional commits (do not force-push during review unless asked
   — it makes the diff history harder to follow).
9. **CI must be green** before merge (currently just `pytest`; this list
   may grow).

## Reporting bugs and requesting features

Open a GitHub Issue. Please include:

- For bugs: minimal reproduction (CLI command + expected vs actual
  output), Python version, OS, and the rate of recurrence.
- For features: the use-case in plain English first, then the proposed
  CLI/API surface. We're fine saying "won't fix" for things that
  conflict with the project's scope (Vietnamese T+N swing trading on
  HOSE/HNX/UPCOM; ACBS fee model; vnstock data source).

For licensing questions (commercial use, CLA, vnstock interaction), open
a GitHub Issue tagged `licensing` rather than emailing — keeps the
conversation public and discoverable for everyone.

## Scope and stability

This is a personal-research project. Breaking changes to the CLI,
ledger schema, or report formats may happen between minor versions.
The 98-test suite is the safety net for regressions in the modeling
math, the trading-calendar arithmetic, and the cache freshness logic;
contributions that break those without obvious cause will not be
merged.
