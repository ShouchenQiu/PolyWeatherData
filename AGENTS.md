# Codex Execution Protocol

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
