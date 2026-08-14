# Agent Notes

An **Agent Note** records a decision that affects this fork — the *why* and *what we
gave up* — the parts code and docs can't carry. Adapted from the
[deepseek-harness convention](https://github.com/deepseek-ai/deepseek-harness/blob/master/.agents/notes/README.md);
we keep its path grammar and file format, and drop the i18n triplets, verification
scripts, and archive freezing until the volume warrants them.

## Layout and naming

`{lifecycle}/{class}/yyyy-mm-dd-topic-title.md`

- **Lifecycle**: `proposed/` (reviewed before building), `implemented/` (shipped; kept
  current with what actually shipped — facts only, never the decision), `rejected/`
  (declined; keep only while the rationale prevents a tempting mistake).
- **Class**: `feature` | `bug-fix` | `simplification` | `architecture` (structure of
  shipped source) | `process` (tooling/workflow around the code) | `testing`.
- Date = when first proposed. Cross-reference other notes with relative markdown links.
- No central index; browse the tree or grep.

## File format

First three lines, exactly:

```markdown
# Agent Note: <title>

Status: <proposed | implemented | rejected — why, in one line>
```

then a blank line, then a body opening with `## Problem` (motivation, standing without
the solution), followed by `## Decision`, `## Alternatives considered`, and
`## Consequences` (plus bespoke technical sections as needed).

A superseded note is never edited into a different decision: write a new note and
cross-link both.
