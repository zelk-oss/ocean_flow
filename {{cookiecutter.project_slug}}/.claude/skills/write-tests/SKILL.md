---
name: write-tests
description: >
  Post-implementation test writing pipeline for this repository. Use
  this skill whenever implementation code already exists and the user
  wants tests written for it — especially when they say "write tests
  for", "add test coverage", "test this module", "cover this with
  tests", "add missing tests", "write tests after implementation", or
  "I just implemented X, now write tests". The skill orchestrates the
  plan-dev and test-writer agents in sequence, with a user approval
  gate between planning and test writing. Target is 100% coverage of
  the tested module. Use it proactively after any non-trivial
  implementation when the user hasn't written tests yet.
---

# write-tests

A four-step pipeline that takes an already-implemented module and
produces a complete test suite targeting 100% coverage. You act as
the orchestrator — you do not write tests yourself.

## Step 1 — Plan

Invoke the `plan-dev` agent. Pass it:
- The name and path of the module or code to test
- Any context the user provided (what the code does, recent changes,
  known edge cases)
- The explicit goal: write tests for existing code, targeting 100%
  coverage of the specified module(s)

The agent will explore the codebase, examine the source files, ask
clarifying questions about scope and coverage expectations, and
return a structured testing plan. Capture the full plan text — you
will present it to the user in step 2.

## Step 2 — User approval

Present the plan to the user using `AskUserQuestion`. Show the
complete plan and ask:

> "Here is the test plan from the planning agent. Does this look
> right to you? Reply 'yes' to proceed, or describe what you'd
> like changed."

If the user requests changes, pass the revised requirements back to
`plan-dev` for another planning round. Repeat until the user
approves. Only move to step 3 once you have explicit approval.

## Step 3 — Write tests

Invoke the `test-writer` agent. Pass it:
- The approved plan (full text)
- The path(s) of the source files to test
- Any clarifications that came up during planning
- Explicit instruction: implementation already exists, tests must
  pass immediately, target 100% coverage

The agent will read the source, write a complete pytest test suite,
run it with coverage, and iterate until the suite is green and
coverage reaches 100%. When it finishes, capture its full report
for step 4.

## Step 4 — Report

Summarise the outcome in a short, structured message to the user:

```
## Test writing complete

**Module tested:** <path to the source module>

**Tests written:** <list of test files created or modified, one per line>

**Test functions added:** <count and brief description of what each
  class covers — e.g. "TestFooFunctional (3), TestFooUnittest (5),
  TestFooErrors (2)">

**Coverage:** <coverage percentage reached; note any lines still
  uncovered and why>

**Bugs found:** <any genuine source bugs discovered; not fixed —
  only reported>

**Next steps:** <open questions or integration scenarios worth
  adding in a future pass>
```

Keep each section to two or three lines. The user can read the
full agent report if they want detail — your job here is a quick,
scannable summary.

## What you never do

- Write tests or source code yourself — delegate entirely to the agents
- Proceed past step 2 without explicit user approval of the plan
- Accept a test suite that is still failing or below 100% coverage
  as "done" without flagging it clearly
- Silently skip the plan-dev step because the module "seems small"
