---
applyTo: "**"
excludeAgent: "coding-agent"
---

# PR Review Standards

## Review Philosophy
* Only comment when you have HIGH CONFIDENCE (>80%) that an issue exists
* Be concise: one sentence per comment when possible
* Focus on actionable feedback, not observations
* When reviewing text, only comment on clarity issues if the text is genuinely confusing or could lead to errors.

## What to Analyze

Review all code changes for:
- Code quality and style consistency with the existing codebase
- Potential bugs or issues
- Performance implications
- Always check for blocking IO in async code
- Security concerns
- Test coverage
- Documentation updates if needed

## Specification
This library implements the [Sendspin spec](https://github.com/Sendspin/spec). When a PR changes protocol behavior (messages, fields, handshake, pairing, timing, or role behavior), read the matching spec file before reviewing: `connection.md`, `pairing.md`, `messaging.md`, or `roles/<family>/v1.md`. If the PR links a spec PR, review against that spec PR instead.
- A message the library sends, or a field it requires, that differs from the spec is a `[CRITICAL]`.
- In `server/`, dropping support for a field the spec renamed or removed, instead of accepting it through `flag_noncompliance`, is a `[PROBLEM]`. The client library supports only the current spec, so dropping old forms there is not an issue.
- Optional spec features the client library does not use yet are not issues.
- Gaps that only a non-compliant client or server can reach are not issues, unless they crash or affect other clients.

## Linked Issues
If the PR references an issue, read it first. Flag as `[CRITICAL]` when the change leaves part of the issue unaddressed, and as `[PROBLEM]` when it patches a symptom or fixes it at the wrong layer.

## Public API
Other projects use this library. Flag as `[PROBLEM]` any removed or renamed public symbol, changed signature, or new abstract method on a store, unless the PR has the `breaking-change` label and its description notes the break. Flag as `[PROBLEM]` features that only make sense for one consumer.

## Existing Review Comments
Flag earlier review comments on the PR that have not been addressed.

## PR Title
The PR title must be a functional description of the change. It must NOT contain conventional commit prefixes such as `feat:`, `fix:`, `refactor:`, `chore:`, etc. Labels are used to categorize PRs, not the title. Flag as `[PROBLEM]` if the title uses such prefixes.

## CI context
**Important**: You review PRs immediately, before CI completes. Do not flag issues that CI will catch.

### What Our CI Checks (`.github/workflows/test.yml`)
**Lint checks:**
* SKIP=no-commit-to-branch pre-commit run --all-files
* See .pre-commit-config.yaml for the full list of checks

**Runs tests:**
* pytest --durations 10 --cov-report term-missing --cov=aiosendspin --cov-report=xml tests/

## Skip These (Low Value)
Do not comment on:

* Style/formatting (pre-commit handles this)
* Test failures
* Missing dependencies (CI handles this)
* Minor naming suggestions
* Suggestions to add comments
* Multiple issues in one comment
* Logging suggestions unless security-related

## Issue Categories
Categorize every issue found as one of:
- `[CRITICAL]` — must be fixed before merging (bugs, security issues, broken functionality)
- `[PROBLEM]` — should be fixed (code quality, bad patterns, missing tests)
- `[SUGGESTION]` — optional improvement (style, minor refactors, nice-to-haves)

## Output
- Post inline comments on GitHub for every `[CRITICAL]` and `[PROBLEM]` issue found.
- Do NOT post `[SUGGESTION]` items to GitHub.
- Do not list things that are already correct

## Output Comment Format
1. State the problem (1 sentence)
2. Why it matters (1 sentence, if needed)
3. Suggested fix (snippet or specific action)

Example:
This could generate a `KeyError` if `"name"` does not exist in the `dict`. Consider using `.get("name")` or adding a check.

## Avoid These False Positives
* Do not report a race unless there is an `await` between the check and the change. Code without an `await` runs without interruption in asyncio, and so does an eager task that never suspends.
* Do not report valid Python 3.12 syntax or features as errors, such as `type X = ...` aliases, `def f[T](...)` type parameters, nested f-strings, and `reversed()` on dictionary views.
* In files that use `from __future__ import annotations`, do not report names used only in annotations as missing at runtime, since those annotations are never evaluated.
* Do not ask for runtime validation of values that the strict type checking already rules out.

## When to Stay Silent
If you're uncertain whether something is an issue, don't comment.
