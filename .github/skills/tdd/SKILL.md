---
name: tdd
description: Test-driven development with red-green-refactor loop. Use when implementing a PolyWeatherBot formal slice, V3 issue/bug/feature task, user-requested TDD/test-first work, or any change that should be driven by observable behavior tests.
---

# Test-Driven Development

## PolyWeatherBot Scope

For this repository, formal slice documents are optional planning inputs, not a
requirement for using this skill:

- If a formal slice exists, use its acceptance criteria, source refs, and
  out-of-scope notes as the approved scope.
- If no formal slice exists, derive one narrow target behavior from the user
  request, failing test, bug report, logs, and current code.

This skill owns only the test-first loop for one target behavior or slice. Do
not re-plan an overhaul, do not implement explicit out-of-scope items, and do
not absorb adjacent deferred work.

## Philosophy

**Core principle**: Tests should verify behavior through public interfaces, not implementation details. Code can change entirely; tests shouldn't.

**Good tests** are integration-style: they exercise real code paths through public APIs. They describe _what_ the system does, not _how_ it does it. A good test reads like a specification - "user can checkout with valid cart" tells you exactly what capability exists. These tests survive refactors because they don't care about internal structure.

**Bad tests** are coupled to implementation. They mock internal collaborators, test private methods, or verify through external means (like querying a database directly instead of using the interface). The warning sign: your test breaks when you refactor, but behavior hasn't changed. If you rename an internal function and tests fail, those tests were testing implementation, not behavior.

See [tests.md](tests.md) for examples and [mocking.md](mocking.md) for mocking guidelines.

## Anti-Pattern: Horizontal Slices

**DO NOT write all tests first, then all implementation.** This is "horizontal slicing" - treating RED as "write all tests" and GREEN as "write all code."

This produces **crap tests**:

- Tests written in bulk test _imagined_ behavior, not _actual_ behavior
- You end up testing the _shape_ of things (data structures, function signatures) rather than user-facing behavior
- Tests become insensitive to real changes - they pass when behavior breaks, fail when behavior is fine
- You outrun your headlights, committing to test structure before understanding the implementation

**Correct approach**: Vertical slices via tracer bullets. One test → one implementation → repeat. Each test responds to what you learned from the previous cycle. Because you just wrote the code, you know exactly what behavior matters and how to verify it.

```
WRONG (horizontal):
  RED:   test1, test2, test3, test4, test5
  GREEN: impl1, impl2, impl3, impl4, impl5

RIGHT (vertical):
  RED→GREEN: test1→impl1
  RED→GREEN: test2→impl2
  RED→GREEN: test3→impl3
  ...
```

## Workflow

### 1. Planning

When exploring the codebase, use the project's domain glossary so that test names and interface vocabulary match the project's language, and respect ADRs in the area you're touching.

If the target behavior is already defined in a formal slice document such as
`docs/issues/proposed-slices*.md`, do not ask the user to reconfirm its
interface or behavior list. If there is no formal slice, infer the smallest
testable behavior from the request and code. Ask only if:

- The target behavior conflicts with current code or tests
- The acceptance criteria are not testable as written
- The work would require live credentials, live network, or human input
- The implementation would touch explicit out-of-scope or deferred work
- A public interface must change in a way the request or slice does not specify

Before writing code:

- [ ] Identify the next acceptance criterion or observable behavior to test
- [ ] Prefer the existing public interface; design for [testability](interface-design.md) only when a new seam is required
- [ ] Use existing tests and fixtures where they fit
- [ ] Choose the smallest relevant `pytest` node to run

### 2. Tracer Bullet

Write ONE test that confirms ONE thing about the system:

```
RED:   Write test for first behavior → test fails
GREEN: Write minimal code to pass → test passes
```

This is your tracer bullet - proves the path works end-to-end.

### 3. Incremental Loop

For each remaining behavior:

```
RED:   Write next test → fails
GREEN: Minimal code to pass → passes
```

Rules:

- One test at a time
- Only enough code to pass current test
- Don't anticipate future tests
- Keep tests focused on observable behavior
- Extend existing tests when they already cover the same public behavior
- Default to `pytest`; run the smallest relevant test node during the loop

### 4. Refactor

After all tests pass, look for [refactor candidates](refactoring.md):

- [ ] Extract duplication
- [ ] Deepen modules (move complexity behind simple interfaces)
- [ ] Apply SOLID principles where natural
- [ ] Consider what new code reveals about existing code
- [ ] Run tests after each refactor step

**Never refactor while RED.** Get to GREEN first.

Keep refactors inside the current target's touched area. If a broader
architecture concern appears, record it for the next architecture review instead
of expanding the work.

If stuck, switch to diagnosis when you have an unclear failure, missing seam,
performance regression, or repeated failed hypothesis.

## Checklist Per Cycle

```
[ ] Test describes behavior, not implementation
[ ] Test uses public interface only
[ ] Test would survive internal refactor
[ ] Code is minimal for this test
[ ] No speculative features added
```

## Completion Checklist

```
[ ] Target behavior or slice acceptance criteria are satisfied
[ ] Target-specific tests pass
[ ] Adjacent tests pass when public interfaces changed
[ ] No explicit out-of-scope or deferred work was implemented
[ ] Residual risk or deferred behavior is noted
```
