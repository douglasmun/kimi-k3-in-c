# Session notes — macOS port

State as of 2026-08-03. Facts that outlived the session they came from, and that the
source alone does not explain. For build and architecture guidance see `CLAUDE.md`.

## What was done

Ported the engine to Apple Silicon (M5 Max, macOS 15, Apple Clang 15). Five source-level
fixes plus Makefile platform detection, and a `macos-14` CI job so the Darwin paths are
verified by CI rather than on one laptop. Full weightless suite passes, including all
three oracle gates.

Also written: `docs/WHY_NOT_QWEN.md`, a case study on why this engine refuses
Qwen3.6-27B — the difference between an inference runtime and an inference
implementation.

## Open items

**PR #4 (`douglasmun:macos-port-only` → upstream `main`) is unmerged.** It is complete,
conflict-free, and has CI evidence linked in a comment. It cannot be merged from this
side: access on `FareedKhan-dev/kimi-k3-in-c` is pull-only, and the API rejects
`MergePullRequest`. It is the maintainer's call.

**Do not delete `macos-port-only` before that merge.** GitHub closes a pull request when
its head branch is deleted. The branch is otherwise redundant — its commits are already
on the fork's `main`.

## What is not verified

**Nothing here has run against real K3 weights.** Correctness is established against the
tiny oracle model in `tests/fixtures` only. The 1.56 TB checkpoint has never been present.

That gap matters most for the Darwin I/O changes, because it is exactly the code the
weightless tests cannot reach: `F_NOCACHE` in place of `O_DIRECT`, and the stubbed
`posix_fadvise`. Both have correct buffered fallbacks, so a failure there degrades
performance rather than results — but the streaming path at real scale is untested on
Darwin. `make test-all SHARD_DIR=...` is the gate to run if a checkpoint ever lands.

## Notes for whoever picks this up

- The tokenizer parity job reports NOT RUN without `tiktoken.model`, which ships with the
  weights. That is deliberate; a check that silently degrades to nothing still shows green.
- Actions on the fork did not fire on push until a `workflow_dispatch` had run once.
  Cause was never pinned down — the permissions API read identical before and after. If
  pushes stop triggering CI, dispatch manually once and see if it resumes.
