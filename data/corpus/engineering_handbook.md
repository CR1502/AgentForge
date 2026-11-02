# Helix Dynamics — Engineering Handbook

This page covers engineering workflow: version control, code review, releases,
and on-call. It is maintained by the Developer Experience pillar.

## Source Control and Branching

All code lives in a single monorepo named **`coil`**, hosted on an internal
GitLab instance. The default branch is `mainline` (not `main` or `master`).
Feature branches must be prefixed with the author's username and a ticket ID,
for example `okoro/REL-1821-retry-logic`.

## Code Review

Every merge request requires approval from **two reviewers**, at least one of
whom must be a designated **code owner** for the affected directory. Merge
requests must pass all CI checks and may not be merged by their own author.

## Release Cadence

Helix Relay follows a **six-week release train**. A release is cut on the first
Monday of each train, spends one week in a staging "bake", and ships to
production the following Monday. Hotfixes bypass the train and require VP
approval.

## Testing Requirements

New code must maintain at least **80% line coverage**. The CI pipeline runs unit
tests, integration tests, and a 30-minute soak test before any production
deploy. The soak test must show zero memory growth.

## On-Call

Each pillar runs a weekly on-call rotation. The primary on-call engineer must
acknowledge a Sev-1 page within **5 minutes**. On-call shifts run Monday to
Monday and come with a **$750** per-week on-call stipend.

## Programming Languages

The Relay data plane is written in **Rust**. Control-plane services are written
in **Go**. Internal tooling and the Atlas frontend use **TypeScript**. Python is
approved only for data analysis and scripts, never for production services.
