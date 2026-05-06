---
name: develop-with-tdd
description: >
  Short test-driven development pipeline for this repository. Use this skill
  whenever the user wants to implement a new feature, fix a bug, or add
  functionality using TDD — especially when they say "develop with TDD",
  "use TDD to implement", "tdd this", "write tests first then implement",
  "run the tdd pipeline", or "implement X with test-driven development".
  The skill orchestrates three project agents in sequence — plan-dev,
  ttd-test-dev, and ttd-code-dev — with a user approval gate between
  planning and test writing. Use it proactively whenever a non-trivial
  feature or fix is requested and the user hasn't specified an approach.
---

# develop-with-tdd

A five-step TDD pipeline that takes a task description from the user and
produces implemented, tested code. You act as the orchestrator — you do
not write tests or source code yourself.

## Step 1 — Plan

Invoke the `plan-dev` agent with the user's task description. Pass the full
context: what the user wants to build, any constraints they mentioned, and
relevant file or module names.

The agent will explore the codebase, ask clarifying questions, and return a
structured plan. Capture the full plan text from the agent's response — you
will present it to the user in step 2.

## Step 2 — User approval

Present the plan to the user using `AskUserQuestion`. Show the complete plan
and ask one of these questions (or a close equivalent):

> "Here is the plan from the planning agent. Does this look right to you?
> Reply 'yes' to proceed, or describe what you'd like changed."

If the user requests changes, update the plan description and pass the
revised requirements back to `plan-dev` for another round. Repeat until the
user approves. Only move to step 3 once you have explicit approval.

## Step 3 — Write failing tests

Invoke the `ttd-test-dev` agent. Pass it:
- The approved plan (full text)
- The user's original task description
- Any clarifications that came up during planning

The agent will write pytest tests that define the expected behaviour. These
tests are intentionally failing — that is correct. When the agent finishes,
save its report for use in step 5.

## Step 4 — Implement

Invoke the `ttd-code-dev` agent. Pass it:
- The approved plan
- The ttd-test-dev report from step 3
- Instruction to write the least viable code that makes the failing tests pass

The agent will read the tests, run the suite, implement source code
iteratively until the suite is green, then return a report. Save this report
for step 5.

## Step 5 — Report

Summarise what was done in a short, structured message to the user:

```
## TDD cycle complete

**Task:** <one-line description of what was implemented>

**Tests written:** <list of new test files or test functions, one per line>

**Source files changed:** <list of created or modified source files>

**Test suite:** <pass/fail counts before → after>

**Key decisions:** <any non-obvious design choices from the implementation report>

**Next steps:** <open questions or recommended follow-ups from the agents>
```

Keep each section to two or three lines. The user can read the full agent
reports if they want detail — your job here is a quick, scannable summary.

## What you never do

- Write tests or source code yourself — delegate entirely to the agents
- Proceed past step 2 without explicit user approval of the plan
- Silently skip the plan-dev step because the task "seems small"
- Merge or paraphrase the agent reports in a way that loses critical detail
  (blockers, assumptions, open questions must surface in step 5)
