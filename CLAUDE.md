# kimi-k3-in-c — working notes

Pure-C, CPU-only inference for Kimi K3 (2.78T MoE). No BLAS, no CUDA, no framework.

This is an inference **implementation**, not a runtime: the architecture is compiled in,
not configured. See `docs/WHY_NOT_QWEN.md` for what that distinction costs and why the
engine refuses other checkpoints rather than adapting to them.

## Read first

`docs/ARCHITECTURE.md` (layer layout, streaming design) · `docs/TESTING.md` (what each
gate proves) · `CONTRIBUTING.md` · `SECURITY.md` (the safetensors parser and the
streaming cache read attacker-influenceable bytes, and are in scope).

## Build and test

```bash
make                  # platform-detected: -mcpu=native on arm64, -march=native on x86
make test && ./bin/k3_model tests/fixtures
```

Everything in `make test` runs **without model weights** — the checkpoint is 1.56 TB, so
correctness has to be verifiable without it. `make test-all SHARD_DIR=...` adds the
checkpoint-dependent tests. The end-to-end gate is `./bin/k3_model tests/fixtures`, which
must print `VERDICT: ENGINE MATCHES THE REFERENCE EXACTLY`; anything less is a failure
even when every unit test passes.

`make portable` targets a documented baseline instead of the local CPU — the right choice
for CI, where the runner's cores are not the user's.

## macOS / Apple Silicon

The port is recent and the Darwin paths are the ones most likely to break.

- **`ru_maxrss` is bytes on Darwin, kilobytes on Linux.** `k3_run.c` scales per-platform.
  Get this wrong and every reported PEAK RSS is off by 1024x — silently, since the number
  still looks plausible. `k3.h` designates those figures authoritative.
- **`_POSIX_C_SOURCE` hides Darwin's BSD extensions**, `F_NOCACHE` and `ru_maxrss` among
  them. `src/io/k3_portable_io.h` sets `_DARWIN_C_SOURCE` and must be included *before*
  any libc header, or the definitions are already gone by the time it lands.
- **No `O_DIRECT` on Darwin.** The shim maps it to `fcntl(fd, F_NOCACHE, 1)`, applied
  after `open()`. Failure is fine: reads stay buffered and correct, just not uncached.
  `posix_fadvise` has no equivalent and is stubbed — it is advisory only.
- **Apple Clang ships no OpenMP runtime.** Needs Homebrew libomp; the Makefile finds it
  via `brew --prefix libomp` and warns if it is missing.

CI covers this: `.github/workflows/ci.yml` runs a `macos-14` job alongside the Ubuntu
matrix. Its first build deliberately passes no arguments, because the defect it guards
against is plain `make` failing on a Mac.

## Conventions

- C99 (`-std=gnu99`), warnings are defects. The engine does pointer arithmetic on
  `const void *` weight pointers, where a missing `-Wpointer-arith` silently strides by
  one byte.
- `-ffp-contract=off` is deliberate. Contraction changes results, and the oracle gates
  compare exactly.
- Fail fast on checkpoint mismatch. A shape the engine did not expect means the config and
  the checkpoint disagree, and every kernel downstream would read wrong strides while
  producing plausible numbers. Refusing loudly is the feature.

## Repo state

`origin` is upstream (`FareedKhan-dev/kimi-k3-in-c`), **pull access only** — no pushing
there, and merging PRs is the maintainer's call. `fork` is `douglasmun/kimi-k3-in-c`,
where work lands. Actions must sometimes be dispatched manually on the fork
(`gh workflow run CI --repo douglasmun/kimi-k3-in-c --ref <branch>`).

No local checkpoint. Nothing here has been run against real K3 weights.

`docs/SESSION_NOTES.md` records the open items — chiefly that PR #4 is unmerged upstream
and that `macos-port-only` must not be deleted before it is.
