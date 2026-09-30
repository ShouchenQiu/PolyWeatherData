# Agent Execution Protocol (historical)

> **Note:** this file is a tooling/process protocol inherited from an earlier,
> much larger version of the repository. It is **not** part of the paper pipeline
> and nothing in `pipeline/` depends on it. It is kept only as a record of the
> execution conventions used while the analysis was being run.

## Long-Running Commands

For this repository, Codex must not directly execute a command expected to run
for more than 5 minutes.

- For a command expected to take 5-30 minutes, provide the exact command to
  the user and wait for them to run it.
- For a command expected to take more than 30 minutes or require persistence,
  provide an exact `tmux` command for the user to run.
- Present `tmux` operations as separate, sequential commands: first create or
  enter the session, then `cd`, create the output directory, set shell options,
  and finally start the Python command. Do not wrap those actions into one
  large shell command or nested `bash -lc` invocation.
- The user runs long commands. Codex only inspects artifacts, logs, and results
  after the user explicitly asks for a check.
- Codex must not start, restart, or attach to a long-running `tmux` job unless
  the user explicitly asks Codex to execute that job.
- Codex must not poll long-running jobs. Do not send periodic status checks,
  repeated `ps`/`tmux`/log commands, or automatic progress updates.

## Required Progress Output

Every command Codex gives the user for a task expected to run longer than five
minutes must emit concise, flushed progress lines and write them to its log.

- Print a `[start]` line with the command stage and immutable output directory.
- Print `[complete]` after every resumable stage, including the relevant row or
  artifact count when available.
- Print a final `[done]` line only after all requested artifacts and the report
  have been written.
- On any error, preserve the Python traceback in the same `tee` log; do not
  suppress stderr or replace failures with a success-looking shell exit.
- For `tmux` commands, use `set -o pipefail` so a failed Python process makes
  the session fail even when its output is piped through `tee`.

When duration is uncertain, state the uncertainty and default to giving the
user the command rather than starting it.

## Goal Protocol

Do not create, continue, update, or complete a Goal unless the user explicitly
asks to use Goal form or Goal execution. Normal implementation work should use
the relevant skills, such as TDD, without a Goal. This prevents autonomous Goal
continuations from conflicting with a long-running command the user is running.

## Experimental Analysis Closure

For every analysis, diagnostic, ablation, or modelling plan, Codex must retain
the concrete questions and mechanism hypotheses stated when the plan was
approved. Completion analysis must answer each item explicitly, not merely
summarize aggregate metrics.

- Report every original question as **supported**, **contradicted**,
  **inconclusive**, or **not measured**, with the relevant artifact,
  stratum/fold evidence, and any sample-size or data-contract limitation.
- Do not infer a cross-fold mechanism from one month, one fold, a descriptive
  aggregate, or a state feature that is constant or unavailable in the panel.
- Separate what the completed experiment establishes from follow-up work still
  required to answer an original question. Name missing data or implementation
  explicitly when a diagnostic was not measured.
- Before proposing the next experiment, use this question-by-question ledger
  to state which uncertainty it resolves and why it is necessary.

## Experiment Intent Before Execution

Before implementing or running a costly experiment, Codex must write a compact
decision ledger from the approved plan: the original decision question, the
required estimand, the data regime, and the artifact that will answer it.

- Do not substitute a stricter, adjacent, or easier estimand for the requested
  one.  In particular, an oracle ceiling, a frozen deployment estimate, and a
  descriptive ablation are different deliverables and must be named and
  reported separately.
- If an experiment needs both an oracle ceiling and a deployment estimate,
  implement and report the ceiling first unless the user explicitly orders the
  reverse.  Do not declare the experiment complete when either required answer
  is missing.
- Before requesting a long run, verify that the runner can produce every
  promised decision artifact.  A probe may validate mechanics, but cannot be
  presented as an answer to an unmeasured estimand.
- At completion, begin with the original decision questions and their answers;
  aggregate metrics and implementation detail are supporting evidence only.
