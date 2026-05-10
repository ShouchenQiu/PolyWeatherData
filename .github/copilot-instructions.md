# PolyWeatherBot -- Copilot Coding Instructions

---

## 1. File Encoding - CRITICAL

**All Python source files (.py) MUST contain only ASCII characters.**

- Never write comments, docstrings, variable names, or string literals in Chinese
  or any other non-ASCII language.
- Use English for all in-code text (comments, docstrings, logging messages,
  exception messages, assert messages).

---

## 2. Read Before Write - CRITICAL

**Always read the full relevant file(s) before making any edit.**

- Never patch a function without first reading the entire module it lives in.
- Never add a parameter or field without checking every call site in the codebase.
- Never introduce a new abstraction without verifying one does not already exist.
- If a design discussion has taken place in the conversation, re-read the affected
  file before coding to confirm the current state matches the assumed baseline.

---

## 3. No Patchwork - CRITICAL

**Each change must be complete and self-consistent. No incremental half-fixes.**

- If fixing a bug requires changing an interface, update ALL call sites in the
  same edit session, not in a follow-up.
- If removing a parameter, remove it from the definition, every call site, and
  every test in one pass.
- If adding a feature, wire it end-to-end (definition + call site + test) before
  stopping. Do not leave stubs that silently do nothing.
- Never add a workaround on top of broken code. Fix the root cause or explicitly
  flag it as a known limitation with a TODO comment.

---

## 4. No Dead Code

**Do not leave unused code in the codebase.**

- If an import, function, class, or variable is no longer referenced after a
  change, delete it in the same edit.
- If an experimental branch of logic has been superseded, remove the old branch
  entirely rather than commenting it out.
- If a feature flag is permanently disabled, remove the guarded code and the flag.

---

## 5. Scope Discipline

**Only change what was asked. Do not silently refactor surrounding code.**

- Do not rename symbols, reformat files, or reorder functions unless explicitly
  requested.
- Do not add logging, error handling, or type annotations to code you did not
  otherwise modify.
- If you notice a separate bug while making a change, report it in the response
  instead of fixing it silently.

---

## 6. Interface Consistency

**Preserve existing interfaces unless a breaking change is explicitly requested.**

- Method signatures, return types, and field names must remain stable across
  edits unless a refactor is the stated goal.
- If a base class or protocol defines an interface, all concrete implementations
  must conform to it exactly.
- When adding a new parameter, give it a default value so existing call sites do
  not break.

---

## 7. Feature Flags via Settings

**Experimental or optional features must be gated by a Settings field.**

- Add a boolean field to `config/settings.py` with a sensible default (usually
  False for new/risky features).
- Read the flag from an environment variable following the existing pattern.
- The feature must be fully skipped when the flag is False; no partial execution.
- Remove the flag and the guard once the feature is promoted to stable.

---

## 8. Layer Boundaries

**Respect the established layer boundaries. Do not reach across layers.**

- `data/` sources must not import from `core/`, `model/`, or `loops/`.
- `model/` must not import from `loops/`, `core/exchange/`, or `snipers/`.
- `loops/` may import from `core/`, `model/`, `data/`, `config/`, and `db/`.
- `core/` must not import from `loops/`, `model/`, or `data/`.
- `config/` must not import from any other project package.
- `db/` must not import from any other project package.
- Circular imports are never acceptable; use `TYPE_CHECKING` guards for
  type-only references across layers.

---

## 9. Wholeness Check Before Submitting

**Before finishing any edit session, verify the following:**

- No import that is now unused was left behind.
- No function that is now unreachable was left behind.
- Every new public symbol has at least one call site or test.
- No TODO was introduced for work that should have been done in this session.
- The change compiles (no syntax errors) and existing tests still pass conceptually.
