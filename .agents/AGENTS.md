# .agents/ — agent workspace

Durable artifacts produced by coding agents working on this fork live here, following the
[deepseek-harness `.agents/` pattern](https://github.com/deepseek-ai/deepseek-harness/tree/master/.agents),
adapted for a small research fork.

Layout:

- **`notes/`** — Agent Notes: decision records and proposals (RFCs written by agents).
  Format and lifecycle rules in [notes/README.md](notes/README.md). Every non-trivial
  change should add or update one in the same commit series.
- There is deliberately **no journal or index**: the working record is carried the
  upstream way — one Agent Note per decision (including retracted or refuted ones,
  with the refutation), `proposed/` notes for future work, dossiers for
  investigations, and git history for the timeline. A session is closed when its
  decisions are noted, not when a diary is written.
- **`debug/`** — investigation dossiers: one dated directory per incident/root-cause
  session (`yyyy-mm-dd-topic/`), holding the report (`report.md`) plus its evidence
  (topology dumps, repro logs). These are agent-to-agent handoff artifacts: a later
  session resumes from the dossier, not from chat history. Frozen once the
  investigation closes; corrections go in dated addenda inside `report.md`.
  *(Our extension — the upstream pattern has no investigation class.)*
- **`skills/`** — reserved for repo-local agent skills (`<name>/SKILL.md`), created
  when the first one lands.

Rules of the road:

- `*.log` and `*.out` under `.agents/` are tracked via **git-lfs** (see `.gitattributes`);
  markdown/XML/JSON stay plain git.
- Upstream (`eunomia-bpf/nccl-eBPF`) uses `docs/tmp/` as its own agents' scratch channel.
  Leave it untouched in this fork — minimizing diff against upstream keeps future
  upstream PRs clean. New agent artifacts go here, never in `docs/tmp/`.
- Nothing in `.agents/` is shipped code or user documentation; user-facing docs live in
  `docs/` (this fork's evaluation docs: `docs/gb300/`).
- **Redaction rule (public fork):** before committing any log, topology dump, or
  report, scrub site identifiers — internal IP addresses, NIC GUIDs, fabric/cluster
  UUIDs, host hashes, usernames. Keep what reproducibility needs (software versions,
  PCI layout, hostnames' tray-number semantics). Verify with:
  `git grep -nE '10\.[0-9]+\.[0-9]+\.[0-9]+|guid="0x|host_hash="0x' -- .agents docs`
  before pushing. Rationale: benchmark artifacts double as reconnaissance material.
